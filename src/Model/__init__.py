import TDSE.tdse as tdse 
import TDSE.tdse_selfmem as tdse_selfmem
import TDSE.tdse_selfmem_spkembmem as tdse_selfmem_spkembmem
import TDSE.tdse_spkembmem  as tdse_spkembmem
import USEV.usev as usev
import USEV.usev_selfmem as usev_selfmem
import SEANet.seanet as seanet
import SEANet.seanet_selfmem as seanet_selfmem
import MoMuSE.momuse as momuse 
import MuSE.muse as muse 

def get_model(model_name: str):
    if model_name == "TDSE":
        return getattr(tdse, "av_convtasnet")
    elif model_name =='TDSE_SelfMem':
        return getattr(tdse_selfmem,"av_convtasnet")
    elif model_name == 'TDSE_SpkEmbMem':
        return getattr(tdse_spkembmem, "av_convtasnet")
    elif model_name =='TDSE_SelfMem_SpkEmbMem':
        return getattr(tdse_selfmem_spkembmem,'av_convtasnet')
    elif model_name=='USEV':
        return getattr(usev,'usev')
    elif model_name=='USEV_SelfMem':
        return getattr(usev_selfmem,'usev')
    elif model_name=='SEANET':
        return getattr(seanet,'seanet')
    elif model_name=='SEANET_SelfMem':
        return getattr(seanet_selfmem,'seanet')
    elif model_name=='MuSE':
        return getattr(muse,'muse')
    elif model_name=='MoMuSE':
        return getattr(momuse,'momuse')
    else:  # model_name error !!!
        print(model_name + " not found !!!")
        exit(1)
    print("load: " + str(model_name) + "!!!")


if __name__ == "__main__":
    model = get_model("TDSE")()
    num_params = sum(param.numel() for param in model.parameters())
