from donut.dashboard_support import Dashboard

# dashboard_plt("../sample_data/8192_7.24.csv", 0.3, None, False)
use_cache_result = True
use_cache_probability = True
dashboard = Dashboard(use_plt=True,
                      train_file="1024_0.csv",
                      test_file="4096_1.88.csv",
                      is_local=True,
                      is_upload=False,
                      src_threshold_value=21.0,
                      a=1,
                      use_cache_result=use_cache_result,
                      use_cache_probability=use_cache_probability)
