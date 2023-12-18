from main import MixturePredictor, GCN
from pairing.data import PairData, Dataset, loader
import pairing.fingerprint
import analysis.auroc
import analysis.best


pred1, y1 = analysis.best.collate_test()
pred2, y2 = pairing.fingerprint.get_test_pred_y()
analysis.auroc.make_dual_chart(pred1,y1,"Our Model",pred2,y2,"Molecular Fingerprints")
