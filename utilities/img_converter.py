import os
from PIL import Image

# Location of sample file
IN_SAMPLE = "../samples/sample_cliff.txt"
# Location of output file from WFC
IN_MAP = "../wfc_out/cliff.txt"
# Directory containing images.  
# NOTE: images are assumed to be .png and must be named with their corresponding UTF-8 character
# (with the exception of "." -> "_dot" and " " -> "_")
IMG_DIR = "./images"
# Directory to save resulting images
OUT_DIR = "."

def main():
    DICT = {}
    tile_w, tile_h = 0, 0
    with open(IN_SAMPLE, encoding="UTF-8") as f:
        # Get dimensions and characters used in input file
        in_rows, in_cols = 0, 0
        for line in f:
            l = line.replace("\n", "")
            print(l)
            if len(l) > in_cols: in_cols = len(l)
            if len(l) > 0: in_rows += 1
            for char in l:
                if char not in DICT: DICT[char] = None
        # Find corresponding images for each symbol and store the size of the tiles
        for path in os.listdir(IMG_DIR):
            if os.path.isfile(os.path.join(IMG_DIR, path)):
                file, ext = path.split(".")
                if file == "_": file = " "
                if file == "_dot": file = "."
                if ext == "png" and file in DICT: 
                    DICT[file] = Image.open(os.path.join(IMG_DIR, path))
                    if DICT[file].width > tile_w: tile_w = DICT[file].width
                    if DICT[file].height > tile_h: tile_h = DICT[file].height
        # print(f"Rows: {in_rows}, Cols: {in_cols}\n")
        # print(DICT)
        f.seek(0)
        # Create a new image and paste in the corresponding tiles for each character
        in_img = Image.new("RGB", (tile_w * in_cols, tile_h * in_rows))
        i = 0
        for line in f:
            j = 0
            for char in line.replace("\n", ""):
                in_img.paste(DICT[char], (j * tile_w, i * tile_h))
                j += 1
            i += 1
        in_img.save(os.path.join(OUT_DIR, "input_img.png"))

    with open(IN_MAP, encoding="UTF-8") as f:
        # Get dimensions of output text file
        out_rows, out_cols = 0, 0
        for line in f:
            l = line.replace("\n", "")
            # print(l)
            if len(l) > out_cols: out_cols = len(l)
            if len(l) > 0: out_rows += 1
        f.seek(0)
        # Create a new image and paste in the corresponding tiles for each character
        out_img = Image.new("RGB", (tile_w * out_cols, tile_h * out_rows))
        i = 0
        for line in f:
            j = 0
            for char in line.replace("\n", ""):
                out_img.paste(DICT[char], (j * tile_w, i * tile_h))
                j += 1
            i += 1
        out_img.save(os.path.join(OUT_DIR, "output_img.png"))

if __name__ == "__main__":
    main()
