from plot_digitizer import PlotAnalysis
from tqdm import tqdm

if __name__ == "__main__":
    pa_engine = PlotAnalysis()
    
    for idx in tqdm(range(len(pa_engine.pd_engine.InsSeg_engine.dataset))):
        try:
            pa_engine.load_img(idx)
            pa_engine.save_best_results( "/home/weixin/Documents/GitProjects/plot_digitizer_BMVC2021/data/output_plot_extraction")
        except:
            print("fail for ", idx)
        