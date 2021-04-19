from donut.data import self_structure


def test_dashboard():
    self_structure(use_plt=True,
                   train_file="4096_14.21.csv",
                   test_file="4096_1.88.csv",
                   is_local=True,
                   is_upload=False,
                   src_threshold_value=None)