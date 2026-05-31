"""
rl_environment_2d_hybrid.py
============================
Hibrit 2B Kutu Paketleme MKS Ortamı

Koordinat tabanlı yerleştirme + Çizge tabanlı durum temsili.

MKS Formülasyonu:
  - Durum: Düğüm özellikleri [w/W, h/H, x/W, y/H, rot, placed] + mekansal çizge
  - Eylem: (nesne_idx, oryantasyon) — sıralama ve döndürme kararı
  - Pozisyon: Bottom-Left (BL) kuralı ile otomatik (heightmap tabanlı)
  - Ödül: Utilization tabanlı veya adım ödülü
  - Terminal: Tüm nesneler yerleştirilmiş veya yerleştirilemez

Çizge yapısı (Fekete-Schepers ile uyumlu):
  - Düğümler: Tüm nesneler (yerleştirilmiş + bekleyen)
  - Kenarlar: Yerleştirilmiş nesneler arası mekansal komşuluk
    - x-komşu: x-izdüşümleri çakışan veya bitişik nesneler
    - y-komşu: y-izdüşümleri çakışan veya bitişik nesneler
"""

import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List


def generate_feasible_2d_items(n_items: int, bin_w: int, bin_h: int,
                                min_size: int = 5,
                                seed: Optional[int] = None) -> List[Tuple[int, int]]:
    """
    Feasible (çözülebilir) dikdörtgen nesneler üret.

    Kutuyu guillotine kesimleriyle bölerek nesneleri oluşturur.
    Bu sayede tüm nesnelerin kutuya sığdığı garanti edilir.

    Yöntem:
    1. Kutuyu (bin_w × bin_h) bir dikdörtgen olarak başlat
    2. Rastgele bir dikdörtgeni yatay veya dikey olarak kes
    3. n_items parçaya ulaşana kadar tekrarla
    4. Parçaları karıştırıp döndür
    """
    rng = np.random.RandomState(seed)
    # Dikdörtgenler listesi: [(w, h), ...]
    rectangles = [(bin_w, bin_h)]

    while len(rectangles) < n_items:
        # Kesilebilecek dikdörtgenleri bul (min_size'dan büyük olanlar)
        cuttable = []
        for idx, (w, h) in enumerate(rectangles):
            if w >= 2 * min_size or h >= 2 * min_size:
                cuttable.append(idx)

        if not cuttable:
            # Daha fazla kesemiyoruz — mevcut parçalarla devam et
            break

        # Rastgele bir dikdörtgen seç (büyük olanları tercih et)
        areas = [rectangles[i][0] * rectangles[i][1] for i in cuttable]
        total_area = sum(areas)
        probs = [a / total_area for a in areas]
        chosen_idx = cuttable[rng.choice(len(cuttable), p=probs)]
        w, h = rectangles[chosen_idx]

        # Yatay mı dikey mi kes?
        can_cut_h = w >= 2 * min_size  # Dikey kesim (genişlikten)
        can_cut_v = h >= 2 * min_size  # Yatay kesim (yükseklikten)

        if can_cut_h and can_cut_v:
            cut_horizontal = rng.random() < 0.5
        elif can_cut_h:
            cut_horizontal = True
        else:
            cut_horizontal = False

        if cut_horizontal:
            # Dikey kesim: genişliği ikiye böl
            cut_pos = rng.randint(min_size, w - min_size + 1)
            piece1 = (cut_pos, h)
            piece2 = (w - cut_pos, h)
        else:
            # Yatay kesim: yüksekliği ikiye böl
            cut_pos = rng.randint(min_size, h - min_size + 1)
            piece1 = (w, cut_pos)
            piece2 = (w, h - cut_pos)

        # Eski dikdörtgeni çıkar, iki yenisini ekle
        rectangles.pop(chosen_idx)
        rectangles.append(piece1)
        rectangles.append(piece2)

    # Fazla parça varsa rastgele seç
    if len(rectangles) > n_items:
        rng.shuffle(rectangles)
        rectangles = rectangles[:n_items]

    # Rastgele döndür (rotasyon çeşitliliği için)
    items = []
    for w, h in rectangles:
        if rng.random() < 0.5:
            items.append((h, w))  # Döndür
        else:
            items.append((w, h))

    rng.shuffle(items)
    return items


def generate_random_2d_items(n_items: int, bin_w: int, bin_h: int,
                              low_ratio: float = 0.1, high_ratio: float = 0.5,
                              seed: Optional[int] = None) -> List[Tuple[int, int]]:
    """
    Rasgele dikdörtgen nesneler üret (eski yöntem, feasible değil).
    Her nesne: (genişlik, yükseklik)
    """
    rng = np.random.RandomState(seed)
    items = []
    for _ in range(n_items):
        w = rng.randint(max(1, int(low_ratio * bin_w)),
                        max(2, int(high_ratio * bin_w)) + 1)
        h = rng.randint(max(1, int(low_ratio * bin_h)),
                        max(2, int(high_ratio * bin_h)) + 1)
        items.append((w, h))
    return items


class BinPacking2DHybridEnv:
    """
    Hibrit 2B-KPP MKS Ortamı.

    Her episode:
    1. N dikdörtgen nesne ile başla
    2. Her adımda ajan bir (nesne, oryantasyon) eylemi seçer
    3. Nesne heightmap'e göre BL pozisyonuna yerleştirilir
    4. Mekansal çizge güncellenir
    5. Tüm nesneler yerleştirilene veya yerleştirilemeyene kadar devam
    """

    def __init__(self, n_items: int = 20, bin_width: int = 100,
                 bin_height: int = 100, reward_type: str = 'step'):
        self.n_items = n_items
        self.bin_width = bin_width
        self.bin_height = bin_height
        self.reward_type = reward_type

        # Nesne bilgileri (orijinal boyutlar)
        self.original_items = []     # [(w, h), ...]

        # Yerleştirme durumu
        self.placed = []             # [bool] — yerleştirilmiş mi?
        self.positions_x = []        # [int] — x koordinatı
        self.positions_y = []        # [int] — y koordinatı
        self.current_w = []          # [int] — mevcut genişlik (rot dikkate alınmış)
        self.current_h = []          # [int] — mevcut yükseklik
        self.rotated = []            # [bool] — döndürülmüş mü?

        # Heightmap (skyline profili)
        self.heightmap = np.zeros(bin_width, dtype=np.int32)

        # Durum
        self.done = False
        self.n_placed = 0
        self.total_item_area = 0
        self.placed_area = 0

        # Çizge
        self.node_features = None    # (N, 6) tensor
        self.adj = None              # (N, N) adjacency

    def reset(self, items: Optional[List[Tuple[int, int]]] = None,
              seed: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Yeni episode başlat."""
        if items is None:
            items = generate_feasible_2d_items(
                self.n_items, self.bin_width, self.bin_height, seed=seed
            )

        n = len(items)
        self.original_items = list(items)
        self.placed = [False] * n
        self.positions_x = [0] * n
        self.positions_y = [0] * n
        self.current_w = [w for w, h in items]
        self.current_h = [h for w, h in items]
        self.rotated = [False] * n

        self.heightmap = np.zeros(self.bin_width, dtype=np.int32)
        self.done = False
        self.n_placed = 0
        self.total_item_area = sum(w * h for w, h in items)
        self.placed_area = 0

        self._rebuild_state()

        if len(self._get_valid_actions()) == 0:
            self.done = True

        return self.get_state()

    def step(self, action_idx: int) -> Tuple[Dict[str, torch.Tensor], float, bool, bool, Dict]:
        """
        Eylem uygula.

        Args:
            action_idx: valid_actions listesindeki indeks

        Returns:
            (state, reward, done, truncated, info)
        """
        if self.done:
            raise RuntimeError("Episode bitti. reset() çağırın.")

        valid_actions = self._get_valid_actions()
        if action_idx < 0 or action_idx >= len(valid_actions):
            raise ValueError(f"Geçersiz action_idx={action_idx}, "
                           f"geçerli aralık: [0, {len(valid_actions)-1}]")

        item_idx, rotation = valid_actions[action_idx]

        # Oryantasyon uygula
        w, h = self.original_items[item_idx]
        if rotation:
            w, h = h, w

        # BL pozisyonu hesapla
        x, y = self._find_bl_position(w, h)
        if x is None:
            # Bu olmamalı çünkü valid_actions kontrol ediyor
            raise RuntimeError(f"Nesne {item_idx} yerleştirilemedi!")

        # Yerleştir
        self.placed[item_idx] = True
        self.positions_x[item_idx] = x
        self.positions_y[item_idx] = y
        self.current_w[item_idx] = w
        self.current_h[item_idx] = h
        self.rotated[item_idx] = rotation
        self.n_placed += 1
        self.placed_area += w * h

        # Heightmap güncelle
        self.heightmap[x:x + w] = np.maximum(
            self.heightmap[x:x + w], y + h
        )

        # Durum güncelle
        self._rebuild_state()

        # Terminal kontrol
        valid_actions_new = self._get_valid_actions()
        if len(valid_actions_new) == 0:
            self.done = True

        # Ödül
        reward = self._compute_reward()

        info = {
            'n_placed': self.n_placed,
            'n_total': len(self.original_items),
            'n_remaining': len(self.original_items) - self.n_placed,
            'n_valid_actions': len(valid_actions_new) if not self.done else 0,
            'utilization': self.get_utilization(),
            'placed_area': self.placed_area,
        }

        return self.get_state(), reward, self.done, False, info

    def get_state(self) -> Dict[str, torch.Tensor]:
        """Mevcut durumu döndür."""
        valid_actions = self._get_valid_actions()

        # Valid actions'ı tensor olarak kodla
        if len(valid_actions) > 0:
            va_tensor = torch.tensor(valid_actions, dtype=torch.long)
        else:
            va_tensor = torch.zeros(0, 2, dtype=torch.long)

        return {
            'node_features': self.node_features.clone(),
            'adj': self.adj.clone(),
            'valid_actions': va_tensor,         # (K, 2) — [item_idx, rotation]
            'heightmap': torch.tensor(self.heightmap, dtype=torch.float32) / self.bin_height,
            'n_nodes': len(self.original_items),
        }

    def get_utilization(self) -> float:
        """Alan kullanım oranı."""
        bin_area = self.bin_width * self.bin_height
        return self.placed_area / bin_area if bin_area > 0 else 0.0

    def get_num_bins(self) -> int:
        """Kalan grup sayısı (yerleştirilemeyen nesneler)."""
        return len(self.original_items) - self.n_placed

    # ---- ÖZEL METOTLAR ----

    def _find_bl_position(self, w: int, h: int) -> Tuple[Optional[int], Optional[int]]:
        """
        Bottom-Left pozisyonu bul.

        Heightmap'te en alçak noktayı arar, nesneyi oraya yerleştirir.

        Returns:
            (x, y) veya (None, None) sığmazsa
        """
        best_x, best_y = None, None
        best_score = float('inf')

        for x in range(self.bin_width - w + 1):
            # Bu x aralığındaki en yüksek nokta
            y = int(np.max(self.heightmap[x:x + w]))

            # Kutuya sığıyor mu?
            if y + h <= self.bin_height:
                # BL kuralı: önce en alçak y, sonra en soldaki x
                score = (y, x)
                if score < (best_score if isinstance(best_score, tuple) else (best_score,)):
                    best_x = x
                    best_y = y
                    best_score = score

        return best_x, best_y

    def _get_valid_actions(self) -> List[Tuple[int, int]]:
        """
        Geçerli eylemleri döndür.

        Her eylem = (item_idx, rotation)
        rotation: 0 = orijinal, 1 = 90° döndürülmüş

        Sadece yerleştirilebilir (sığan) eylemler dahil.
        """
        actions = []
        for i, (w, h) in enumerate(self.original_items):
            if self.placed[i]:
                continue

            # Orijinal oryantasyon
            x, y = self._find_bl_position(w, h)
            if x is not None:
                actions.append((i, 0))

            # Döndürülmüş (w != h ise farklı)
            if w != h:
                x2, y2 = self._find_bl_position(h, w)
                if x2 is not None:
                    actions.append((i, 1))

        return actions

    def _rebuild_state(self):
        """Düğüm özellikleri ve çizgeyi yeniden oluştur."""
        n = len(self.original_items)
        W = self.bin_width
        H = self.bin_height

        # Düğüm özellikleri: [w/W, h/H, x/W, y/H, rot, placed]
        feat = torch.zeros(n, 6)
        for i in range(n):
            if self.placed[i]:
                feat[i, 0] = self.current_w[i] / W
                feat[i, 1] = self.current_h[i] / H
                feat[i, 2] = self.positions_x[i] / W
                feat[i, 3] = self.positions_y[i] / H
                feat[i, 4] = 1.0 if self.rotated[i] else 0.0
                feat[i, 5] = 1.0
            else:
                # Bekleyen nesne: orijinal boyutlar, konum 0
                w, h = self.original_items[i]
                feat[i, 0] = w / W
                feat[i, 1] = h / H
                feat[i, 2] = 0.0
                feat[i, 3] = 0.0
                feat[i, 4] = 0.0
                feat[i, 5] = 0.0

        self.node_features = feat

        # Mekansal çizge: yerleştirilmiş nesneler arası kenarlar
        adj = torch.zeros(n, n)
        placed_indices = [i for i in range(n) if self.placed[i]]

        for idx_a in range(len(placed_indices)):
            for idx_b in range(idx_a + 1, len(placed_indices)):
                i = placed_indices[idx_a]
                j = placed_indices[idx_b]

                if self._are_spatially_adjacent(i, j):
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0

        self.adj = adj

    def _are_spatially_adjacent(self, i: int, j: int) -> bool:
        """
        İki yerleştirilmiş nesne mekansal olarak komşu mu?

        Komşuluk: x veya y izdüşümlerinde çakışma/bitişiklik varsa.
        Bu, Fekete-Schepers'ın Gx/Gy kenarlarına karşılık gelir.
        """
        xi, yi = self.positions_x[i], self.positions_y[i]
        wi, hi = self.current_w[i], self.current_h[i]
        xj, yj = self.positions_x[j], self.positions_y[j]
        wj, hj = self.current_w[j], self.current_h[j]

        # x-izdüşüm çakışması: [xi, xi+wi) ∩ [xj, xj+wj) ≠ ∅
        x_overlap = (xi < xj + wj) and (xj < xi + wi)

        # y-izdüşüm çakışması: [yi, yi+hi) ∩ [yj, yj+hj) ≠ ∅
        y_overlap = (yi < yj + hj) and (yj < yi + hi)

        # Bitişiklik (tam temas)
        x_adjacent = (xi + wi == xj) or (xj + wj == xi)
        y_adjacent = (yi + hi == yj) or (yj + hj == yi)

        # Komşu = (x çakışma VE y bitişik) VEYA (y çakışma VE x bitişik)
        # → fiziksel olarak temas eden nesneler
        return (x_overlap and y_adjacent) or (y_overlap and x_adjacent)

    def _compute_reward(self) -> float:
        """Ödül hesapla."""
        if self.reward_type == 'step':
            # Yerleştirilen nesnenin alan oranına dayalı ödül
            if self.n_placed > 0:
                last_idx = self._last_placed()
                item_area = self.current_w[last_idx] * self.current_h[last_idx]
                bin_area = self.bin_width * self.bin_height
                reward = item_area / bin_area

                # Tüm nesneler yerleştirildi — bonus!
                if self.done and self.n_placed == len(self.original_items):
                    reward += 2.0
                # Episode bitti ama nesneler kaldı — ceza
                elif self.done and self.n_placed < len(self.original_items):
                    remaining = len(self.original_items) - self.n_placed
                    reward -= 0.1 * remaining

                return reward
            return 0.0
        elif self.reward_type == 'terminal':
            if self.done:
                return self.get_utilization()
            return 0.0
        elif self.reward_type == 'utilization_step':
            # Her adımda eklenen alanın oranı
            if self.n_placed > 0 and not self.done:
                last_area = self.current_w[self._last_placed()] * self.current_h[self._last_placed()]
                return last_area / (self.bin_width * self.bin_height)
            return 0.0
        else:
            raise ValueError(f"Bilinmeyen ödül tipi: {self.reward_type}")

    def _last_placed(self) -> int:
        """Son yerleştirilen nesnenin indeksini döndür."""
        for i in range(len(self.placed) - 1, -1, -1):
            if self.placed[i]:
                return i
        return -1

    def render(self):
        """Mevcut durumu yazdır."""
        print(f"\n{'='*60}")
        print(f"Yerleşen: {self.n_placed}/{len(self.original_items)} | "
              f"Done: {self.done} | Util: {self.get_utilization():.1%}")
        print(f"Kutu: {self.bin_width}×{self.bin_height}")

        for i in range(len(self.original_items)):
            ow, oh = self.original_items[i]
            if self.placed[i]:
                rot_str = " ROT" if self.rotated[i] else ""
                print(f"  [{i}] {ow}×{oh}{rot_str} → "
                      f"({self.positions_x[i]},{self.positions_y[i]}) "
                      f"{self.current_w[i]}×{self.current_h[i]}")
            else:
                print(f"  [{i}] {ow}×{oh} — bekleyen")

        valid = self._get_valid_actions()
        print(f"Geçerli eylemler: {len(valid)}")
        for idx, (item_i, rot) in enumerate(valid[:10]):
            w, h = self.original_items[item_i]
            if rot:
                w, h = h, w
            x, y = self._find_bl_position(w, h)
            print(f"  [{idx}] nesne={item_i} rot={rot} → "
                  f"({x},{y}) {w}×{h}")
        if len(valid) > 10:
            print(f"  ... ve {len(valid)-10} daha")

        # Heightmap görselleştirme
        max_h = int(np.max(self.heightmap)) if np.max(self.heightmap) > 0 else 1
        print(f"\nHeightmap (max={max_h}):")
        for row in range(min(max_h + 2, self.bin_height), -1, -1):
            line = ""
            for col in range(self.bin_width):
                if self.heightmap[col] > row:
                    # Bu hücre dolu — hangi nesne?
                    char = '█'
                    for i in range(len(self.original_items)):
                        if self.placed[i]:
                            if (self.positions_x[i] <= col < self.positions_x[i] + self.current_w[i] and
                                self.positions_y[i] <= row < self.positions_y[i] + self.current_h[i]):
                                char = chr(65 + i)  # A, B, C, ...
                                break
                    line += char
                else:
                    line += '·'
            print(f"  {row:2d}|{line}|")
        print(f"    +{'─'*self.bin_width}+")
        print(f"{'='*60}")


# ---- TEST ----
if __name__ == '__main__':
    print("=" * 60)
    print("Hibrit 2B Kutu Paketleme Ortamı Testi")
    print("=" * 60)

    # Örnekteki nesneler
    items = [(4, 3), (3, 5), (5, 2), (2, 4)]
    env = BinPacking2DHybridEnv(n_items=4, bin_width=10, bin_height=10,
                                 reward_type='step')

    # --- Test 1: Rotasyonsuz, A→B→C→D sırasıyla ---
    print("\n>>> Test 1: Rotasyonsuz, sıralı yerleştirme")
    state = env.reset(items=items)
    env.render()

    # A(4×3) orijinal, B(3×5) orijinal, C(5×2) orijinal, D(2×4) orijinal
    actions_no_rot = []
    valid = env._get_valid_actions()
    for target_item, target_rot in [(0, 0), (1, 0), (2, 0), (3, 0)]:
        for idx, (item_i, rot) in enumerate(valid):
            if item_i == target_item and rot == target_rot:
                actions_no_rot.append(idx)
                break
        state, reward, done, _, info = env.step(actions_no_rot[-1])
        valid = env._get_valid_actions()

    env.render()
    print(f"Utilization: {env.get_utilization():.1%}")
    print(f"Max heightmap: {np.max(env.heightmap)}")

    # --- Test 2: Rotasyonlu, akıllı sıralama (B_rot→A→C→D_rot) ---
    print("\n>>> Test 2: Rotasyonlu, akıllı sıralama")
    state = env.reset(items=items)

    # B(3×5)→rot→5×3, A(4×3)→orijinal, C(5×2)→orijinal, D(2×4)→rot→4×2
    sequence = [(1, 1), (0, 0), (2, 0), (3, 1)]
    for target_item, target_rot in sequence:
        valid = env._get_valid_actions()
        for idx, (item_i, rot) in enumerate(valid):
            if item_i == target_item and rot == target_rot:
                state, reward, done, _, info = env.step(idx)
                break

    env.render()
    print(f"Utilization: {env.get_utilization():.1%}")
    print(f"Max heightmap: {np.max(env.heightmap)}")

    # --- Test 3: Rasgele 20 nesne ---
    print("\n>>> Test 3: 20 rasgele nesne, greedy yerleştirme")
    env2 = BinPacking2DHybridEnv(n_items=20, bin_width=100, bin_height=100)
    state = env2.reset(seed=42)
    print(f"Nesneler: {len(env2.original_items)}, Geçerli eylemler: {len(env2._get_valid_actions())}")

    step = 0
    while not env2.done:
        state, reward, done, _, info = env2.step(0)  # Her zaman ilk eylemi seç
        step += 1

    print(f"Adımlar: {step}, Yerleşen: {env2.n_placed}/{len(env2.original_items)}, "
          f"Utilization: {env2.get_utilization():.1%}")
    print(f"Çizge kenar sayısı: {int(env2.adj.sum().item()) // 2}")
