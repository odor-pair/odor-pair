from main import MixturePredictor, GCN
import pairing.data
import numpy as np
import torchmetrics
import analysis.best

import matplotlib.pyplot as plt

COUNT_PER_ROW = 26
# If we have less than this number of datapoints,
# we italicize
TRAIN_LIM = 10
TEST_LIM = 2


def get_score(pred, y):
    auroc = torchmetrics.classification.MultilabelAUROC(y.shape[1],average=None)
    return np.mean(auroc(pred,y.int()).numpy())

def make_score_dict(pred,y,covered_notes):
    auroc = torchmetrics.classification.MultilabelAUROC(len(covered_notes),average=None)
    scores = auroc(pred,y.int()).numpy()
    score_dict = dict(zip(covered_notes,scores))

    microauroc = torchmetrics.classification.MultilabelAUROC(len(covered_notes),average='micro')
    score_dict["TOTAL"] = microauroc(pred,y.int()).numpy().item()
    return score_dict


# Sorted based on first input
def make_dual_chart(title,pred1,y1,label1,pred2,y2,label2,notes):
    all_notes = np.array(notes)
    auroc = torchmetrics.classification.MultilabelAUROC(y1.shape[1],average=None)
    scores1 = auroc(pred1,y1.int()).numpy()
    scores2 = auroc(pred2,y2.int()).numpy()
    print(label1,make_score_dict(pred1,y1,all_notes))
    print(label2,make_score_dict(pred2,y2,all_notes))
    exit()

    
    idcs = np.flip(np.argsort(scores1))
    scores1 = scores1[idcs]
    scores2 = scores2[idcs]
    all_notes = all_notes[idcs]
    
    idxs = [i for i in range(len(all_notes))]

    w = .4

    plt.figure(figsize=(15, 3))
    plt.bar(idxs,scores1,width=w,align='edge')
    plt.bar([i+w for i in idxs],scores2,width=w,align='edge')
    plt.legend([f"{label1} (AUROC={np.mean(scores1):.2f})",f"{label2} (AUROC={np.mean(scores2):.2f})"])
    plt.axhline(y=0.5,color='grey',linestyle='dashed')
    plt.xticks(ticks=idxs,labels=all_notes,rotation=45)
    plt.title(title)
    plt.tight_layout()

    for ticklabel in plt.gca().get_xticklabels():
        idx = ticklabel._x
        if scores1[idx] < .5 or scores1[idx] < scores2[idx]:
            ticklabel.set_color('r')
        else:
            ticklabel.set_color('black')

    plt.show()

def make_chart(pred,y):
    all_notes = np.array(pairing.data.get_all_notes())
    auroc = torchmetrics.classification.MultilabelAUROC(Dataset.num_classes(),average=None)
    scores = auroc(pred,y.int()).numpy()

    idcs = np.flip(np.argsort(scores))
    scores = scores[idcs]
    all_notes = all_notes[idcs]

    plt.figure(figsize=(15, 3))
    plt.bar(all_notes,scores,align='edge')
    plt.axhline(y=0.5,color='grey',linestyle='dashed')
    plt.xticks(rotation=45)
    plt.title("AUROC by Odor Label")
    plt.tight_layout()
    plt.show()

# Sort based on the first passeed in dictionary
def make_chart_from_dictionary(all_note_to_scores, all_model_names, train_frequencies, test_frequencies):
    assert len(all_note_to_scores) == len(all_model_names)

    num_models = len(all_model_names)

    all_notes = [np.array(list(note_to_score.keys())) for note_to_score in all_note_to_scores]
    all_notes_sets = [set(notes) for notes in all_notes]
    notes = np.array(list(set.intersection(*all_notes_sets)))
    all_scores = [np.array([note_to_score[n] for n in notes]) for note_to_score in all_note_to_scores]
    
    bad_notes = set()

    for n,f in train_frequencies.items():
        if f < TRAIN_LIM:
            bad_notes.add(n)

    for n,f in test_frequencies.items():
        if f < TEST_LIM:
            bad_notes.add(n)

    sort_index = np.flip(np.argsort(all_scores[0]))
    all_scores = [scores[sort_index] for scores in all_scores]
    
    notes = notes[sort_index]

    idxs = [i for i in range(len(notes))]

    # Slight margin between bars
    w = (1 / num_models) - .1

    num_rows = int(len(notes)/COUNT_PER_ROW) + 1
    f,axs = plt.subplots(num_rows,figsize=(12, 2.5*num_rows))
    for i in range(num_rows):
        sidx = i * COUNT_PER_ROW
        eidx = (i + 1) * COUNT_PER_ROW

        for midx in range(num_models):
            bar_idxs = [i+w*midx for i in idxs]
            axs[i].bar(bar_idxs[sidx:eidx],all_scores[midx][sidx:eidx],width=w,align='edge',label=all_model_names[midx])

        axs[i].set_xticks(ticks=idxs[sidx:eidx],labels=notes[sidx:eidx])
        axs[i].axhline(y=0.5,color='grey',linestyle='dashed')
        axs[i].tick_params(axis='x', rotation=45)
        axs[i].set_ylim(0,1)


        for ticklabel in axs[i].get_xticklabels():
            note = ticklabel.get_text() 
            if note in bad_notes:
                print(note,train_frequencies[note])
                ticklabel.set_fontweight('bold')


    # Get the handles from any axis
    handles, _ = axs[-1].get_legend_handles_labels()
    labels = [f"{model_name} (AUROC={np.mean(all_scores[i]):.2f})" for i,model_name in enumerate(all_model_names)]
    legend = plt.figlegend(handles, labels, loc='upper right')

    plt.suptitle("AUROC Comparison on Blended Pair Task by Model and Odor Label")
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.75)
    plt.show()

if __name__ == "__main__":
    pred, y = analysis.best.collate_test()
    make_chart(pred,y)