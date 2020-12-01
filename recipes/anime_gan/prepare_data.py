import os


def test_face_crop():
    from PIL import Image
    import animeface
    im = Image.open('data/test.png')
    faces = animeface.detect(im)
    x, y, w, h = faces[0].face.pos
    im = im.crop((x, y, x + w, y + h))
    im.show()  # display


def list_files(fld_path):
    res = []
    for (_, dirnames, _) in os.walk(fld_path):
        print(f"Count fld: {len(dirnames)}")
        for dirname in dirnames:
            for (_, _, fnames) in os.walk(os.path.join(fld_path, dirname)):
                res.extend([os.path.join(dirname, fname) for fname in fnames if any(ext in fname for ext in ['jpg', 'png', 'JPG'])])
        break
    assert all(fname[fname.rindex('.') + 1:] in ['jpg', 'png'] for fname in res)

    return res


def create_listing(fld_name):
    res = list_files(fld_name)
    print(res[:4], res[-3:])
    out_fname = os.path.join(fld_name, "listing.txt")
    with open(out_fname, 'w') as f:
        f.write("\n".join(res))
    print(f"Dumped listing to {out_fname}")
    return out_fname


if __name__ == '__main__':
    store_fld = "data"
    dirnames = ["danbooru", "animeface-character-dataset"]
    fnames = list(map(create_listing, [os.path.join(store_fld, dname) for dname in dirnames]))
    final_listing = os.path.join(store_fld, "listing.txt")
    with open(final_listing, "w") as out:
        for fname in fnames:
            with open(fname, 'r') as f:
                for row in f:
                    out.write(os.path.join(os.path.basename(os.path.dirname(fname)), row))
