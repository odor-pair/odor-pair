from main import MixturePredictor, GCN
import single.embedding
import single.fingerprint
import analysis.auroc


pred1,y1 = single.embedding.get_train_pred_y()
pred2,y2 = single.fingerprint.get_train_pred_y()

analysis.auroc.make_dual_chart(pred1,y1,"Our Model",pred2,y2,"Molecular Fingerprints")
