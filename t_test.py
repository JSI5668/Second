import scipy.stats
import numpy as np

# ## Kitti performance #############################################################################################
#
# proposed_method = [56.52, 54.65]
# second_best_method = [53.72, 53.91]
# # third_best_method = [0.12, 0.46]
#
# lresult = scipy.stats.levene(proposed_method, second_best_method)
# print('LeveneResult(F) : %.3f \np-value : %.3f' % (lresult))
# m_miou = scipy.stats.ttest_ind(proposed_method, second_best_method, equal_var=False)
# print("m_miou", m_miou)
# # m_miou_rel = scipy.stats.ttest_rel(proposed_method, second_best_method)
# # print("m_miou_rel", m_miou_rel)
# print('#############################################################################################')
# ########################################################################
#
# proposed_method_overall_acc = [87.51, 87.03]
# second_best_method_overall_acc = [86.70, 87.01]
# # third_best_method = [0.12, 0.46]
#
# lresult_overall_acc = scipy.stats.levene(proposed_method_overall_acc, second_best_method_overall_acc)
# print('LeveneResult(F) : %.3f \np-value : %.3f' % (lresult_overall_acc))
# m_overall_acc = scipy.stats.ttest_ind(proposed_method_overall_acc, second_best_method_overall_acc, equal_var=False)
# print("m_overall_acc", m_overall_acc)
# # m_overall_acc_rel = scipy.stats.ttest_rel(proposed_method_overall_acc, second_best_method_overall_acc)
# # print("m_overall_acc_rel", m_overall_acc_rel)
# print('#############################################################################################')
# ########################################################################
#
# proposed_method_mean_acc = [64.97, 62.15]
# second_best_method_mean_acc = [61.78, 62.31]
# #
# lresult_mean_acc = scipy.stats.levene(proposed_method_mean_acc, second_best_method_mean_acc)
# print('LeveneResult(F) : %.3f \np-value : %.3f' % (lresult_mean_acc))
# m_mean_acc = scipy.stats.ttest_ind(proposed_method_mean_acc, second_best_method_mean_acc, equal_var=False)
# print("m_mean_acc", m_mean_acc)
# # m_mean_acc_rel = scipy.stats.ttest_rel(proposed_method_mean_acc, second_best_method_mean_acc)
# # print("m_overall_acc_rel", m_mean_acc_rel)
# print('#############################################################################################')
# ########################################################################
#
# proposed_method_Freq_iou = [78.31, 77.37]
# second_best_method_Freq_iou = [77.08, 77.30]
# # third_best_method = [0.12, 0.46]
#
# m_Freq_iou = scipy.stats.ttest_ind(proposed_method_Freq_iou, second_best_method_Freq_iou, equal_var=False)
# print("m_Freq_iou", m_Freq_iou)
# print('#############################################################################################')
#


## Camvid performance #############################################################################################

proposed_method = [69.48, 69.68]
second_best_method = [68.59, 68.65]
# third_best_method = [0.12, 0.46]

lresult = scipy.stats.levene(proposed_method, second_best_method)
print('LeveneResult(F) : %.3f \np-value : %.3f' % (lresult))
m_miou = scipy.stats.ttest_ind(proposed_method, second_best_method, equal_var=False)
print("m_miou", m_miou)
# m_miou_rel = scipy.stats.ttest_rel(proposed_method, second_best_method)
# print("m_miou_rel", m_miou_rel)
print('#############################################################################################')
########################################################################

# proposed_method_overall_acc = [92.84, 92.94]
# second_best_method_overall_acc = [92.75, 92.85]
# # third_best_method = [0.12, 0.46]
#
# lresult_overall_acc = scipy.stats.levene(proposed_method_overall_acc, second_best_method_overall_acc)
# print('LeveneResult(F) : %.3f \np-value : %.3f' % (lresult_overall_acc))
# m_overall_acc = scipy.stats.ttest_ind(proposed_method_overall_acc, second_best_method_overall_acc, equal_var=False)
# print("m_overall_acc", m_overall_acc)
# # m_overall_acc_rel = scipy.stats.ttest_rel(proposed_method_overall_acc, second_best_method_overall_acc)
# # print("m_overall_acc_rel", m_overall_acc_rel)
# print('#############################################################################################')
########################################################################

# proposed_method_mean_acc = [76.67, 76.96]
# second_best_method_mean_acc = [75.76, 75.94]
# #
# lresult_mean_acc = scipy.stats.levene(proposed_method_mean_acc, second_best_method_mean_acc)
# print('LeveneResult(F) : %.3f \np-value : %.3f' % (lresult_mean_acc))
# m_mean_acc = scipy.stats.ttest_ind(proposed_method_mean_acc, second_best_method_mean_acc, equal_var=False)
# print("m_mean_acc", m_mean_acc)
# # m_mean_acc_rel = scipy.stats.ttest_rel(proposed_method_mean_acc, second_best_method_mean_acc)
# # print("m_overall_acc_rel", m_mean_acc_rel)
print('#############################################################################################')
########################################################################
#
# proposed_method_Freq_iou = [87.04, 87.17]
# second_best_method_Freq_iou = [86.90, 87.05]
# # third_best_method = [0.12, 0.46]
#
# m_Freq_iou = scipy.stats.ttest_ind(proposed_method_Freq_iou, second_best_method_Freq_iou, equal_var=False)
# print("m_Freq_iou", m_Freq_iou)
# print('#############################################################################################')

# ###cohen's d value
# t = m_miou.statistic
# df = len(proposed_method) + len(second_best_method) - 2
# print(abs(t) / np.sqrt(df)), print("d-value of m_miou")
#
# t_mean_acc = m_mean_acc.statistic
# df = len(proposed_method_mean_acc) + len(second_best_method_mean_acc) - 2
# print(abs(t_mean_acc) / np.sqrt(df)), print("d-value of m_mean_acc")