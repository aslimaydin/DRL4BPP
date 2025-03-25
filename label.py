# print("3000 3000")
# j = 0
# n = 5
# x = 378
# y = 200
# x0 = 0
# y0 = 0
# for i in range(75):
#    if i % n == 0 and i != 0:
#        j += 1

#    print(f"{x} {y} {x0 + (x * (i % n))} {y0 + (y * j)}")

# j = 0
# n = 2
# x = 555
# y = 496
# x0 = 1890
# y0 = 0
# for i in range(10):
#    if i % n == 0 and i != 0:
#        j += 1

#    print(f"{x} {y} {x0 + (x * (i % n))} {y0 + (y * j)}")
   


n_sets = int(input("Number of sets (int):"))

for set_idx in range(n_sets):
    bin_width = float(input(f"\n\nBin {set_idx+1} width (float) (0 to skip):"))
    if bin_width == 0:
        continue

    bin_height = float(input(f"Bin {set_idx+1} height (float):"))

    n_items = int(input("Number of items (int):"))

    items_info = []
    for item_idx in range(n_items):
        item_width = float(input(f"\n    Item {item_idx+1} width (float):"))

        item_height = float(input(f"    Item {item_idx+1} height (float):"))
        item_x = float(input(f"\n    Item {item_idx+1} x position (float):"))
        item_y = float(input(f"    Item {item_idx+1} y position (float):"))

        items_info.append([item_width, item_height, item_x, item_y])
    
    with open(f"data/gcut/gcut{set_idx+1}.txt", "w") as f:
        f.write(f"{bin_width} {bin_height}\n")
        for item_info in items_info:
            f.write(f"{item_info[0]} {item_info[1]} {item_info[2]} {item_info[3]}\n")
    
    print("Write successful\n\n")
