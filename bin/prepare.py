from pnia.models.nlam import create_mesh 
from pnia.datasets import SmeagolDataset




if __name__ == "__main__": 

    dataset = SmeagolDataset.from_json("/home/mrpa/chabotv/pnia/pnia/xp_conf/smeagol.json")
    create_mesh.prepare(dataset=dataset, hierarchical=True)