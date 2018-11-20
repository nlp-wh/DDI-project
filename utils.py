import os
import platform
import subprocess


# Save the best result
def save_best_result(class_name, cur_f1_score, result_dir):
    # 0. Replace numpy.float64 to native python type
    cur_f1_score = float(cur_f1_score)
    # result - previous_f1 - MCPCNN, CNN
    # result - best_result - MCPCNN - model.json, weight.h5
    best_result_dir = os.path.join(result_dir, 'best_result')
    previous_f1_dir = os.path.join(result_dir, 'previous_f1')
    model_file_path = os.path.join(result_dir, 'model.json')
    weight_file_path = os.path.join(result_dir, 'weights.h5')
    # 1. Make the directory to save the best result, previous_f1_idr
    if not os.path.exists(best_result_dir):
        os.mkdir(best_result_dir)
    if not os.path.exists(previous_f1_dir):
        os.mkdir(previous_f1_dir)
    if not os.path.exists(os.path.join(best_result_dir, class_name)):
        os.mkdir(os.path.join(best_result_dir, class_name))
    # 2. Get the information of the previous result
    # 3. If info doesn't exist, make new file. And save the best result
    previous_score_file_path = os.path.join(previous_f1_dir, class_name)
    if not os.path.exists(previous_score_file_path):
        with open(previous_score_file_path, 'w', encoding='utf-8') as f:
            f.write(str(cur_f1_score))
            # Save the best result, copy the result
            if platform.platform().lower().startswith('windows'):
                subprocess.call(['copy', '.\\' + model_file_path, '.\\' + os.path.join(best_result_dir, class_name), '/y'], shell=True)
                subprocess.call(['copy', '.\\' + weight_file_path, '.\\' + os.path.join(best_result_dir, class_name), '/y'], shell=True)
            else:
                subprocess.call(['cp', model_file_path, os.path.join(best_result_dir, class_name)])
                subprocess.call(['cp', weight_file_path, os.path.join(best_result_dir, class_name)])
    else:
        # 4. If the current result is better than before, save the result
        #    then, update the result on f1 result file
        f = open(previous_score_file_path, 'r', encoding='utf-8')
        pre_f1_score = float(f.read().strip())
        f.close()
        if pre_f1_score < cur_f1_score:
            with open(previous_score_file_path, 'w', encoding='utf-8') as f:
                f.write(str(cur_f1_score))
                # Save the best result, copy the result
                if platform.platform().lower().startswith('windows'):
                    subprocess.call(['copy', '.\\' + model_file_path, '.\\' + os.path.join(best_result_dir, class_name), '/y'], shell=True)
                    subprocess.call(['copy', '.\\' + weight_file_path, '.\\' + os.path.join(best_result_dir, class_name), '/y'], shell=True)
                else:
                    subprocess.call(['cp', model_file_path, os.path.join(best_result_dir, class_name)])
                    subprocess.call(['cp', weight_file_path, os.path.join(best_result_dir, class_name)])
