from donut.dashboard_support import Dashboard

use_cache_result = False
use_cache_probability = True
use_plt = False
dashboard = Dashboard(use_plt=use_plt,
                      train_file="1024_1.csv",
                      test_file="1024_2.csv",
                      is_local=True,
                      is_upload=False,
                      src_threshold_value=None,
                      a=1,
                      use_cache_result=use_cache_result,
                      use_cache_probability=use_cache_probability)
