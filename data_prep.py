from loader import crossval

src_fold = "./datasource"
init_fold = "./data_init"
kfolds = "./data"

crossval(src_fold, init_fold, first_split=True)
crossval(init_fold+"/train", kfolds)

