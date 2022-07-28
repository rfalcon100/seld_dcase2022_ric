import os
import numpy as np
if __name__ == "__main__":
    from dcase2022_metrics import parameters, cls_compute_seld_results, SELD_evaluation_metrics
else:
    from evaluation.dcase2022_metrics import parameters, cls_compute_seld_results, SELD_evaluation_metrics


def get_accdoa_labels(accdoa_in, nb_classes):
    x, y, z = accdoa_in[:, :, :nb_classes], accdoa_in[:, :, nb_classes:2 * nb_classes], accdoa_in[:, :, 2 * nb_classes:]
    sed = np.sqrt(x ** 2 + y ** 2 + z ** 2) > 0.5

    return sed, accdoa_in

def get_multi_accdoa_labels(accdoa_in, nb_classes):
    """
    Args:
        accdoa_in:  [batch_size, frames, num_track*num_axis*num_class=3*3*12]
        nb_classes: scalar
    Return:
        sedX:       [batch_size, frames, num_class=12]
        doaX:       [batch_size, frames, num_axis*num_class=3*12]
    """
    x0, y0, z0 = accdoa_in[:, :, :1*nb_classes], accdoa_in[:, :, 1*nb_classes:2*nb_classes], accdoa_in[:, :, 2*nb_classes:3*nb_classes]
    sed0 = np.sqrt(x0**2 + y0**2 + z0**2) > 0.5
    doa0 = accdoa_in[:, :, :3*nb_classes]

    x1, y1, z1 = accdoa_in[:, :, 3*nb_classes:4*nb_classes], accdoa_in[:, :, 4*nb_classes:5*nb_classes], accdoa_in[:, :, 5*nb_classes:6*nb_classes]
    sed1 = np.sqrt(x1**2 + y1**2 + z1**2) > 0.5
    doa1 = accdoa_in[:, :, 3*nb_classes: 6*nb_classes]

    x2, y2, z2 = accdoa_in[:, :, 6*nb_classes:7*nb_classes], accdoa_in[:, :, 7*nb_classes:8*nb_classes], accdoa_in[:, :, 8*nb_classes:]
    sed2 = np.sqrt(x2**2 + y2**2 + z2**2) > 0.5
    doa2 = accdoa_in[:, :, 6*nb_classes:]

    return sed0, doa0, sed1, doa1, sed2, doa2

def determine_similar_location(sed_pred0, sed_pred1, doa_pred0, doa_pred1, class_cnt, thresh_unify, nb_classes):
    if (sed_pred0 == 1) and (sed_pred1 == 1):
        if SELD_evaluation_metrics.distance_between_cartesian_coordinates(doa_pred0[class_cnt], doa_pred0[class_cnt+1*nb_classes], doa_pred0[class_cnt+2*nb_classes],
                                                  doa_pred1[class_cnt], doa_pred1[class_cnt+1*nb_classes], doa_pred1[class_cnt+2*nb_classes]) < thresh_unify:
            return 1
        else:
            return 0
    else:
        return 0

def write_output_format_file(_output_format_file, _output_format_dict, use_cartesian=True, ignore_src_id=False):
    """
    Writes DCASE output format csv file, given output format dictionary

    :param _output_format_file:
    :param _output_format_dict:
    :return:
    """
    _fid = open(_output_format_file, 'w')
    # _fid.write('{},{},{},{}\n'.format('frame number with 20ms hop (int)', 'class index (int)', 'azimuth angle (int)', 'elevation angle (int)'))
    for _frame_ind in _output_format_dict.keys():
        for _value in _output_format_dict[_frame_ind]:
            if use_cartesian:
                # Write Cartesian format output. Since baseline does not estimate track count we use a fixed value.
                _fid.write('{},{},{},{},{},{}\n'.format(int(_frame_ind), int(_value[0]), 0, float(_value[1]), float(_value[2]), float(_value[3])))
            else:
                if ignore_src_id:
                    _fid.write(
                        '{},{},{},{}\n'.format(int(_frame_ind), int(_value[0]), round(_value[1]), round(_value[2])))
                else:
                    _fid.write(
                        '{},{},{},{},{}\n'.format(int(_frame_ind), int(_value[0]), 0, round(_value[1]), round(_value[2])))
    _fid.close()

def all_seld_eval(args, directory_root, fnames, pred_directory, result_path=None):
    fpaths = [os.path.join(directory_root, x) for x in fnames]
    fpaths = [os.path.dirname(x.replace('./', '')) for x in fpaths]
    fpaths = [os.path.dirname(x.replace("foa", "metadata").replace("mic", "metadata")) for x in fpaths]
    ref_desc_files = fpaths[0]
    pred_output_format_files = pred_directory

    params = parameters.get_params()
    score_obj = cls_compute_seld_results.ComputeSELDResults(params, ref_files_folder=ref_desc_files)

    #seld_metric = score_obj.get_SELD_Results(pred_output_format_files)
    seld_metric_macro, seld_metric_micro = score_obj.get_SELD_Results(pred_output_format_files, num_classes=args.unique_classes)

    return seld_metric_macro, seld_metric_micro

if __name__ == "__main__":
    pass
