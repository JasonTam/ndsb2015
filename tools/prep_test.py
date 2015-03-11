from tools import my_io

OUT_SHAPE = (64, 64)



my_io.multi_extract(im_files,
                    db_path,
                    backend='lmdb',
                    perturb=True,
                    out_shape=
                    OUT_SHAPE,
                    transfer_feats=True,
                    verbose=False)