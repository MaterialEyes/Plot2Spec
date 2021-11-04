from plot_digitizer import PlotDigitizer, PlotAnalysis


import os


def run(pd_engine, idx, save_results="./"):
    pd_engine.prediction_from_ins_seg(idx)
    pd_engine.flow_prep()
    pd_engine.flow(posi_id=0, 
                   grad_threshold=1., 
                   mode="optical_flow_with_seg_gray_grads_posi_color_correction_grad_rejection_adaptive_dfs",  
                   momentum=True,
                   LK_method=False)
    if not os.path.exists(save_results):
        os.makedirs(save_results)
    pd_engine.save_results(save_results)
    
    
    
    
    
if __name__ == "__main__":
    pd_engine = PlotDigitizer()
    pd_engine.visual_dict["result"]["momentum_factor"] = 0.35
    pd_engine.visual_dict["result"]["dfs_threshold"] = 0.83
    pd_engine.visual_dict["result"]["neighbor_size"] = 5
    pd_engine.visual_dict["result"]["color_threshold"] = 0.8
    pd_engine.visual_dict["result"]["overlap_threshold"] = 0.
    pd_engine.visual_dict["result"]["keep_order_threshold"] = 0.
    pd_engine.visual_dict["result"]["gradient_rejection"] = 20
    pd_engine.visual_dict["result"]["dfs_factor"] = 4.
    pd_engine.visual_dict["result"]["max_dfs_threshold"] = 0.95
    
    for idx in range(100):
        run(pd_engine, idx, "./pd_results/")
    
    