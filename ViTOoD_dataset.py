# -*- coding: utf-8 -*-

import json, os


DATA_INC = "imagenet-C"
DATA_INP = "imagenet-P"

DATA_INP = "imagenet-D"
DATA_INA = "imagenet-A"
DATA_INO = "imagenet-O"
DATA_INR = "imagenet-A"
DATA_INS = "imagenet-sketch"
DATA_INAST = "stylized-imagenet"

DATA_ALL = [DATA_INC, DATA_INP, DATA_INP,DATA_INA,DATA_INO,DATA_INR,DATA_INS,DATA_INAST]
KEYS_CLEAN = ["clean"]

KEYS_CC = ["Brightness", "Contrast", "Defocus Blur", "Elastic Transform", 
           "Fog", "Frost", "Gaussian Noise", "Glass Blur", "Impulse Noise",
           "Jpeg Compression", "Motion Blur", "Pixelate", "Shot Noise", 
           "Snow", "Zoom Blur"]

KEYS_CP = ["Brightness", "Gaussian Noise", "Motion Blur", "Shot Noise", 
           "Snow", "Zoom blur", "Scale", "Rotate", "Tilf", "Translate"]

KEYS_D = ['texture','background','material']
KEYS_ALL = KEYS_CLEAN + KEYS_CC + KEYS_CP+KEYS_D

class VitOoDDataset:
    """
    Helper class to query evaluation results.
    
    Attributes
    ----------
    keys_clean : list
        key for evaluation results on clean data: ["clean"] 

    keys_cc : list
        list that contains keys for all corruption C types evaluated
    keys_cd : list
        list that contains keys for all IN-D types evaluated
    keys_cp : list
        list that contains keys for all corruption P types evaluated
    keys_all : list
        list that contains all keys
    data_inC : str = "imagenet-C"
    data_inP : str = "imagenet-P"
    data_inA : str = "imagenet-A"
    data_inO : str = "imagenet-O"
    data_inR : str = "imagenet-R"
    data_inD : str = "imagenet-D"
    data_inS : str = "imagenet-sketch"
    data_imagenetstyle : str = "stylized-imagenet"
    data : list
        list that contains all data sources ["imagenet-C", "imagenet-P", "imagenet-A","imagenet-O","imagenet-R","imagenet-D","imagenet-sketch","stylized-imagenet"]
    """
    
    keys_clean = KEYS_CLEAN
    keys_cc = KEYS_CC
    keys_cp = KEYS_CP
    keys_D = KEYS_D
    keys_all = KEYS_ALL
    keys_cc = KEYS_CC
    keys_all = KEYS_ALL
    data_inC = DATA_INC
    data_inP = DATA_INP
    data_inA = DATA_INA
    data_inO = DATA_INO
    data_inR = DATA_INR
    data_inD = DATA_INC
    data_inS = DATA_INS
    data_inST = DATA_INAST
    data = DATA_ALL
    
    ############################################################################
    def __init__(self, path="VitOoD-data"):
        """
        Parameters
        ----------
        path : str
            Path to the root folder of the dataset data.
        """
        
        self.path = path
        with open(path) as f:
            self.meta = json.load(f)
    
    ############################################################################
    def _ensure_list(self, l):
        if type(l) is not list:
            l = [l]
        return l
    
    ############################################################################
    def query(
            self,
            data = DATA_ALL,
            key = KEYS_ALL,
            measure = ["accuracy", "aurra"],
            level = 5,
            missing_ok = False,
            tqdm = None
        ):
        """
        Query evaluation results.
        Returns a dictionary: dict[<data>][<architecture id>][<corruption>][<measure type>]
        
        Parameters
        ----------
        data : str/list
            Data used for evaluation.
        key : str/list
            Type data or corruption type.
        measure : str/list
            Measure type ("accuracy", "aurra")
        """
        
        data = self._ensure_list(data)
        key = self._ensure_list(key)
        measure = self._ensure_list(measure)
        
        pbar = tqdm.tqdm(
            total = len(data)*len(key)*len(measure)
        ) if tqdm is not None else None
            
        result = {d:{k:{} for k in self.meta.keys()} for d in data}

        # print(result)
        for d in data:
            
            if d != VitOoDDataset.data_inC and d != VitOoDDataset.data_inP and d != VitOoDDataset.data_inD :
                q = d
                if d == VitOoDDataset.data_inS:
                    q = 'sketch'
                if d == VitOoDDataset.data_inST:
                    q = 'stylized-imagenet'

                for id,value in self.meta.items():
                    if id == "1000":
                        break

                    print(value)
                    print(d)
                    result[d][id][q] = value['performance']['Imagenet'][q]
                    result[d][id]["net_setting"] = value['net_setting']
                    
                    if pbar is not None:
                        pbar.update(1)
            else:
                
                if d == VitOoDDataset.data_inC:
                    for k in key: 
                        if k in VitOoDDataset.keys_cc:
                            for id,value in self.meta.items():
                                if id == "1000":
                                    break
                                get_levels = {}
                                get_levels[level] = value['performance']['Imagenet']['corruption'][k][level]
                                result[d][id][k] = get_levels
                                result[d][id]["net_setting"] = value['net_setting']
                                if pbar is not None:
                                    pbar.update(1)


                elif d == VitOoDDataset.data_inP:
                    for k in key: 
                        if k in VitOoDDataset.keys_cp:
                            for id,value in self.meta.items():  
                                if id == "1000":
                                    break
                                result[d][id][k] = value['performance']['Imagenet']['corruption_P'][k]
                                result[d][id]["net_setting"] = value['net_setting']
                                if pbar is not None:
                                    pbar.update(1)

                elif d == VitOoDDataset.data_inD:
                    for k in key: 
                        if k in VitOoDDataset.keys_D:
                            for id,value in self.meta.items():  
                                if id == "1000":
                                    break
                                result[d][id][k] = value['performance']['Imagenet'][d][k]
                                result[d][id]["net_setting"] = value['net_setting']
                                if pbar is not None:
                                    pbar.update(1)
            if pbar is not None:
                pbar.close()
        
        return result
    
    
    ############################################################################
    def id_to_arch_setting(self, i):
        """
        Returns the Architectures setting representing an architecture in AutoFormer Search Space for the given id.
        
        Parameters
        ----------
        i : str/int
            Architecture id.
        """
        
        return self.meta[str(i)]["net_setting"]
    
    ############################################################################
    

