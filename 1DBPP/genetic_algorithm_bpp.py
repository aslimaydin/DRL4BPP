"""
genetic_algorithm_bpp.py
========================
1B-KPP icin Grouping Genetic Algorithm (GGA) uygulamasi.
Falkenauer (1996) tarafindan onerilen GGA-CGT yaklasiminin basitlestirilmis versiyonu.

Basvuru formundaki basari olcutu:
  - "Yontemlerin metasezgisel yontemlerden, ayni calisma suresi icerisinde, ortalama %30 daha iyi sonuc vermesi"
  - "Metasezgisel yontemlere kiyasla, ayni performans degerlerine ulasma surelerinde %50 iyilestirme"

Kullanim:
    from genetic_algorithm_bpp import solve_bpp_ga
    n_bins, assignment = solve_bpp_ga(weights, capacity, time_limit=5.0)
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import time
import copy
from typing import List, Tuple, Optional, Dict


# ---------------------------------------------------------------
# SEZGISEL YARDIMCI FONKSIYONLAR
# ---------------------------------------------------------------

def first_fit_decreasing(weights: List[int], capacity: int) -> List[List[int]]:
    """FFD ile baslangic cozumu olustur. Kutulari liste olarak dondurur."""
    sorted_indices = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)
    bins = []  # Her bin: [item_idx, ...]
    bin_remaining = []  # Her bin'in kalan kapasitesi

    for idx in sorted_indices:
        w = weights[idx]
        placed = False
        for b in range(len(bins)):
            if bin_remaining[b] >= w:
                bins[b].append(idx)
                bin_remaining[b] -= w
                placed = True
                break
        if not placed:
            bins.append([idx])
            bin_remaining.append(capacity - w)

    return bins


def best_fit_decreasing(weights: List[int], capacity: int) -> List[List[int]]:
    """BFD ile baslangic cozumu olustur."""
    sorted_indices = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)
    bins = []
    bin_remaining = []

    for idx in sorted_indices:
        w = weights[idx]
        best_bin = -1
        min_remaining = capacity + 1

        for b in range(len(bins)):
            if bin_remaining[b] >= w and bin_remaining[b] - w < min_remaining:
                min_remaining = bin_remaining[b] - w
                best_bin = b

        if best_bin >= 0:
            bins[best_bin].append(idx)
            bin_remaining[best_bin] -= w
        else:
            bins.append([idx])
            bin_remaining.append(capacity - w)

    return bins


# ---------------------------------------------------------------
# BIREY (INDIVIDUAL) SINIFI
# ---------------------------------------------------------------

class Individual:
    """GA bireyi: Kutularin listesi olarak temsil edilir (group encoding)."""

    def __init__(self, bins: List[List[int]], weights: List[int], capacity: int):
        self.bins = [list(b) for b in bins]  # Derin kopya
        self.weights = weights
        self.capacity = capacity
        self._fitness = None

    @property
    def n_bins(self) -> int:
        return len(self.bins)

    @property
    def fitness(self) -> float:
        """Fitness: Kutu kullanim orani ortalamasi (yuksek = iyi)."""
        if self._fitness is None:
            if not self.bins:
                self._fitness = 0.0
            else:
                utilizations = []
                for b in self.bins:
                    total = sum(self.weights[i] for i in b)
                    utilizations.append(total / self.capacity)
                self._fitness = sum(u ** 2 for u in utilizations) / len(self.bins)
        return self._fitness

    def invalidate(self):
        """Fitness cache'i sifirla."""
        self._fitness = None

    def is_valid(self) -> bool:
        """Cozumun gecerli olup olmadigini kontrol et."""
        all_items = set()
        for b in self.bins:
            total = sum(self.weights[i] for i in b)
            if total > self.capacity:
                return False
            for i in b:
                if i in all_items:
                    return False
                all_items.add(i)
        return len(all_items) == len(self.weights)

    def copy(self) -> 'Individual':
        ind = Individual(self.bins, self.weights, self.capacity)
        ind._fitness = self._fitness
        return ind


# ---------------------------------------------------------------
# GENETIK OPERATORLER
# ---------------------------------------------------------------

def crossover_gga(parent1: Individual, parent2: Individual,
                  rng: np.random.Generator) -> Individual:
    """
    GGA Caprazlama (Crossover):
    1. Parent1'den en iyi kutulari sec
    2. Parent2'den kalan itemleri tamamla
    """
    weights = parent1.weights
    capacity = parent1.capacity

    # Parent1'in kutularini fitness'a gore sirala (en dolu kutu = en iyi)
    bin_utils = []
    for b in parent1.bins:
        total = sum(weights[i] for i in b)
        bin_utils.append((total / capacity, b))
    bin_utils.sort(reverse=True)

    # En iyi kutularin yarisini sec
    n_keep = max(1, len(bin_utils) // 2)
    child_bins = [list(b) for _, b in bin_utils[:n_keep]]
    placed_items = set()
    for b in child_bins:
        for i in b:
            placed_items.add(i)

    # Kalan itemleri parent2'nin kutularindan al
    remaining = [i for i in range(len(weights)) if i not in placed_items]

    if remaining:
        # Parent2'nin kutu sirasina gore yerles
        p2_assignment = {}
        for b_idx, b in enumerate(parent2.bins):
            for i in b:
                p2_assignment[i] = b_idx

        # Parent2 kutu sirasina gore grupla
        p2_groups = {}
        for i in remaining:
            b_idx = p2_assignment.get(i, -1)
            if b_idx not in p2_groups:
                p2_groups[b_idx] = []
            p2_groups[b_idx].append(i)

        # Gruplari child'a ekle, kapasite kontrolu ile
        for items in p2_groups.values():
            # BFD mantigi ile yerlestir
            for item in sorted(items, key=lambda x: weights[x], reverse=True):
                w = weights[item]
                best_bin = -1
                min_rem = capacity + 1

                for b_idx, b in enumerate(child_bins):
                    cur = sum(weights[j] for j in b)
                    rem = capacity - cur
                    if rem >= w and rem - w < min_rem:
                        min_rem = rem - w
                        best_bin = b_idx

                if best_bin >= 0:
                    child_bins[best_bin].append(item)
                else:
                    child_bins.append([item])

    # Bos kutulari kaldir
    child_bins = [b for b in child_bins if b]

    return Individual(child_bins, weights, capacity)


def mutate_swap(individual: Individual, rng: np.random.Generator) -> Individual:
    """Mutasyon: Iki farkli kutudan birer item sec ve yer degistir."""
    ind = individual.copy()

    if len(ind.bins) < 2:
        return ind

    # Iki farkli kutu sec
    b1, b2 = rng.choice(len(ind.bins), size=2, replace=False)

    if not ind.bins[b1] or not ind.bins[b2]:
        return ind

    # Birer item sec
    i1 = rng.choice(len(ind.bins[b1]))
    i2 = rng.choice(len(ind.bins[b2]))

    item1 = ind.bins[b1][i1]
    item2 = ind.bins[b2][i2]

    # Kapasite kontrolu
    w1 = ind.weights[item1]
    w2 = ind.weights[item2]

    bin1_total = sum(ind.weights[x] for x in ind.bins[b1])
    bin2_total = sum(ind.weights[x] for x in ind.bins[b2])

    new_bin1_total = bin1_total - w1 + w2
    new_bin2_total = bin2_total - w2 + w1

    if new_bin1_total <= ind.capacity and new_bin2_total <= ind.capacity:
        ind.bins[b1][i1] = item2
        ind.bins[b2][i2] = item1
        ind.invalidate()

    return ind


def mutate_move(individual: Individual, rng: np.random.Generator) -> Individual:
    """Mutasyon: Bir itemin baska bir kutuya tasir, bos kalan kutuyu siler."""
    ind = individual.copy()

    if len(ind.bins) < 2:
        return ind

    # Kaynak kutu sec
    src = rng.integers(len(ind.bins))
    if not ind.bins[src]:
        return ind

    # Item sec
    item_idx = rng.integers(len(ind.bins[src]))
    item = ind.bins[src][item_idx]
    w = ind.weights[item]

    # Hedef kutu bul (en iyi uyan)
    best_dst = -1
    min_rem = ind.capacity + 1

    for b in range(len(ind.bins)):
        if b == src:
            continue
        total = sum(ind.weights[x] for x in ind.bins[b])
        rem = ind.capacity - total
        if rem >= w and rem - w < min_rem:
            min_rem = rem - w
            best_dst = b

    if best_dst >= 0:
        ind.bins[src].pop(item_idx)
        ind.bins[best_dst].append(item)

        # Bos kutuyu sil
        if not ind.bins[src]:
            ind.bins.pop(src)

        ind.invalidate()

    return ind


# ---------------------------------------------------------------
# ANA GA FONKSIYONU
# ---------------------------------------------------------------

def solve_bpp_ga(weights: List[int], capacity: int,
                 pop_size: int = 50,
                 n_generations: int = 500,
                 mutation_rate: float = 0.3,
                 elite_ratio: float = 0.1,
                 time_limit: float = 10.0,
                 seed: Optional[int] = None,
                 verbose: bool = False) -> Tuple[int, List[List[int]]]:
    """
    Grouping Genetic Algorithm ile 1B-KPP cozer.

    Args:
        weights: Nesne agirliklari
        capacity: Kutu kapasitesi
        pop_size: Populasyon buyuklugu
        n_generations: Maksimum jenerasyon sayisi
        mutation_rate: Mutasyon orani
        elite_ratio: Elit birey orani
        time_limit: Maksimum calisma suresi (saniye)
        seed: Random seed
        verbose: Detayli cikti

    Returns:
        (n_bins, bins): Kutu sayisi ve kutu atamalari
    """
    rng = np.random.default_rng(seed)
    start_time = time.time()
    n_items = len(weights)

    # Baslangic populasyonu olustur
    population = []

    # FFD ile bir birey
    ffd_bins = first_fit_decreasing(weights, capacity)
    population.append(Individual(ffd_bins, weights, capacity))

    # BFD ile bir birey
    bfd_bins = best_fit_decreasing(weights, capacity)
    population.append(Individual(bfd_bins, weights, capacity))

    # Rastgele permutasyonlarla FF uygulayarak geri kalanini olustur
    for _ in range(pop_size - 2):
        perm = rng.permutation(n_items)
        bins = []
        bin_remaining = []

        for idx in perm:
            w = weights[idx]
            placed = False
            for b in range(len(bins)):
                if bin_remaining[b] >= w:
                    bins[b].append(idx)
                    bin_remaining[b] -= w
                    placed = True
                    break
            if not placed:
                bins.append([idx])
                bin_remaining.append(capacity - w)

        population.append(Individual(bins, weights, capacity))

    # En iyi cozum
    best = min(population, key=lambda ind: ind.n_bins)
    best_n_bins = best.n_bins

    n_elite = max(1, int(pop_size * elite_ratio))

    for gen in range(n_generations):
        # Zaman kontrolu
        if time.time() - start_time > time_limit:
            break

        # Fitness'a gore sirala (dusuk kutu = iyi, yuksek fitness = iyi)
        population.sort(key=lambda ind: (-ind.fitness, ind.n_bins))

        # Elit bireyler
        new_population = [population[i].copy() for i in range(n_elite)]

        # Caprazlama ve mutasyon
        while len(new_population) < pop_size:
            # Turnuva secimi
            candidates = rng.choice(len(population), size=3, replace=False)
            p1 = min(candidates, key=lambda i: population[i].n_bins)

            candidates = rng.choice(len(population), size=3, replace=False)
            p2 = min(candidates, key=lambda i: population[i].n_bins)

            # Caprazlama
            child = crossover_gga(population[p1], population[p2], rng)

            # Mutasyon
            if rng.random() < mutation_rate:
                if rng.random() < 0.5:
                    child = mutate_move(child, rng)
                else:
                    child = mutate_swap(child, rng)

            new_population.append(child)

        population = new_population

        # En iyiyi guncelle
        gen_best = min(population, key=lambda ind: ind.n_bins)
        if gen_best.n_bins < best_n_bins:
            best = gen_best.copy()
            best_n_bins = best.n_bins
            if verbose:
                elapsed = time.time() - start_time
                print(f"  Gen {gen:4d}: {best_n_bins} bins (t={elapsed:.1f}s)")

    elapsed = time.time() - start_time
    if verbose:
        print(f"  GA sonuc: {best_n_bins} bins, {elapsed:.1f}s, {gen+1} jenerasyon")

    return best_n_bins, best.bins


# ---------------------------------------------------------------
# TEST
# ---------------------------------------------------------------

if __name__ == '__main__':
    import json
    from bpplib_loader import load_dataset, load_optimal_solutions
    from rl_utils import first_fit_decreasing as ffd_util, best_fit_decreasing as bfd_util

    print("=" * 70)
    print("Genetik Algoritma (GGA) Benchmark Testi")
    print("=" * 70)

    solutions = load_optimal_solutions(r'bpplib_data\Instances\Solutions\Solutions.xlsx')

    # Scholl_1 N<=50 ile hizli test
    instances = load_dataset('scholl_1', max_items=50)
    print(f"\nScholl_1 (N<=50): {len(instances)} ornek")

    ga_results = []
    ffd_results = []
    ga_times = []

    for inst in instances:
        # FFD
        ffd_n, _ = ffd_util(inst['weights'], inst['capacity'])
        ffd_results.append(ffd_n)

        # GA
        start = time.time()
        ga_n, _ = solve_bpp_ga(inst['weights'], inst['capacity'],
                               time_limit=2.0, seed=42)
        ga_time = time.time() - start
        ga_results.append(ga_n)
        ga_times.append(ga_time)

    opt_list = [solutions.get(inst['name'], {}).get('ub', None) for inst in instances]
    valid_opts = [o for o in opt_list if o]

    print(f"\n  Optimum ort: {np.mean(valid_opts):.1f}")
    print(f"  FFD ort:     {np.mean(ffd_results):.1f}")
    print(f"  GA ort:      {np.mean(ga_results):.1f}")
    print(f"  GA sure:     {np.mean(ga_times):.2f}s/ornek")

    ga_opt = sum(1 for g, o in zip(ga_results, opt_list) if o and g == o)
    ffd_opt = sum(1 for g, o in zip(ffd_results, opt_list) if o and g == o)
    print(f"  FFD optimum: {ffd_opt}/{len(instances)}")
    print(f"  GA optimum:  {ga_opt}/{len(instances)}")
