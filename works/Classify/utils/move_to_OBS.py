"""
华为云modelarts
将modelarts文件传到obs桶中
"""
# import moxing
#
#
# moxing.file.copy_parallel(src_url='../customize_service.py', dst_url='obs://lance/food/om/customize_service.py')


from modelarts.session import Session
session = Session()
# session.obs.download_file(src_obs_file="obs://ruyiwei/train-part1.zip", dst_local_dir="/home/ma-user/work/Classify/dataset/train-part1.zip")
# # session.obs.download_file(src_obs_file="obs://lance/road/vali_20k", dst_local_dir="/home/ma-user/work/Classify/dataset/vali_20k")
# session.obs.download_file(src_obs_file="obs://ruyiwei/test_50k.zip", dst_local_dir="/home/ma-user/work/Classify/dataset/test_50k.zip")
session.obs.download_file(src_obs_file="obs://ruyiwei/train-part2.zip", dst_local_dir="/home/ma-user/work/Classify/dataset/train-part2.zip")
session.obs.download_file(src_obs_file="obs://ruyiwei/vali_20k.zip", dst_local_dir="/home/ma-user/work/Classify/dataset/vali_20k.zip")


