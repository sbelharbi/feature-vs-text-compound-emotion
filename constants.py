# Datasets
MELD = 'MELD'
C_EXPR_DB = 'C-EXPR-DB'  # test set of AWAW 7th, compound
# emotion, annotated by our team and split into 125 videos.
C_EXPR_DB_CHALLENGE = 'C-EXPR-DB-CHALLENGE'  # test set of AWAW 7th, compound
# emotion

DATASETS = [MELD, C_EXPR_DB, C_EXPR_DB_CHALLENGE]

NUM_CLASSES = {
    MELD: 7,
    C_EXPR_DB: 7,
    C_EXPR_DB_CHALLENGE: 7
}

# Task
CLASSFICATION = 'CLASSIFICATION'
REGRESSION = 'REGRESSION'

TASKS = [CLASSFICATION, REGRESSION]

DS_TASK = {
    MELD: CLASSFICATION,
    C_EXPR_DB: CLASSFICATION,
    C_EXPR_DB_CHALLENGE: CLASSFICATION
}

# Fusion methods:
LFAN = 'LFAN'  # https://arxiv.org/pdf/2203.13031
CAN = 'CAN'  # https://arxiv.org/pdf/2203.13031
JMT = 'JMT'  # https://arxiv.org/pdf/2403.10488
MT = 'MT'  # https://arxiv.org/pdf/2403.10488

FUSION_METHODS = [LFAN, CAN, JMT, MT]

# optimizers
SGD = 'SGD'
ADAM = 'ADAM'

OPTIMIZERS = [SGD, ADAM]


# LR scheduler
STEP = 'STEP'
MULTISTEP = 'MULTISTEP'
MYSTEP = 'MYSTEP'
MYWARMUP = 'MYWARMUP'
COSINE = 'COSINE'
MYCOSINE = 'MYCOSINE'

LR_SCHEDULERS = [STEP, MULTISTEP, MYSTEP, MYWARMUP, COSINE, MYCOSINE]

# Mode for lr scheduler
MAX_MODE = 'MAX'
MIN_MODE = 'MIN'

LR_MODES = [MAX_MODE, MIN_MODE]

# Modes
TRAINING = "TRAINING"
EVALUATION = 'EVALUATION'

MODES = [TRAINING, EVALUATION]


CROP_SIZE = 224
RESIZE_SIZE = 256

SZ224 = 224
SZ256 = 256
SZ112 = 112


# EXPRESSIONS
# basic
SURPRISE = 'Surprise'
FEAR = 'Fear'
DISGUST = 'Disgust'
HAPPINESS = 'Happiness'
SADNESS = 'Sadness'
ANGER = 'Anger'
NEUTRAL = 'Neutral'

# compound
FEARFULLY_SURPRISED = "Fearfully Surprised"
HAPPILY_SURPRISED = "Happily Surprised"
SADLY_SURPRISED = "Sadly Surprised"
DISGUSTEDLY_SURPRISED = "Disgustedly Surprised"
ANGRILY_SURPRISED = "Angrily Surprised"
SADLY_FEARFUL = "Sadly Fearful"
SADLY_ANGRY = "Sadly Angry"
OTHER = "Other"


EXPRESSIONS = [SURPRISE, FEAR, DISGUST, SADNESS, HAPPINESS, ANGER, NEUTRAL,
               FEARFULLY_SURPRISED,
               HAPPILY_SURPRISED,
               SADLY_SURPRISED,
               DISGUSTEDLY_SURPRISED,
               ANGRILY_SURPRISED,
               SADLY_FEARFUL,
               SADLY_ANGRY,
               OTHER
               ]

# dataset splits:
TRAINSET = 'train'
VALIDSET = 'val'
TESTSET = 'test'

SPLITS = [TRAINSET, VALIDSET, TESTSET]

# Data modalities
VGGISH = 'vggish'  # audio: vggish features
VIDEO = 'video'  # video: row frames
BERT = 'bert'  # text: BERT features
EXPR = 'EXPR_continuous_label'

MODALITIES = [VGGISH, VIDEO, BERT, EXPR]

# Metrics: macro F1 score
MACRO_F1 = 'MACRO_F1'  # Macro F1: unweighted average.
W_F1 = 'W_F1'  # F1 score: weighted.
CL_ACC = 'CL_ACC'  # classification acc.
CFUSE_MARIX = 'CONFUSION_MATRIX'  # confusion matrix

METRICS = [MACRO_F1, W_F1, CL_ACC, CFUSE_MARIX]

# levels
FRAME_LEVEL = 'FRAME_LEVEL'  # frame level
VIDEO_LEVEL = 'VIDEO_LEVEL'  # video level

EVAL_LEVELS = [FRAME_LEVEL, VIDEO_LEVEL]

# how to predict at video level from frame prediction
FRM_VOTE = 'FRAMES_VOTE'  # video label = majority voting at frame level
FRM_AVG_PROBS = 'FRAMES_AVG_PROBS'  # compute probs of each frame,
# then average them, then predict for video.
FRM_AVG_LOGITS = 'FRAMES_AVG_LOGITS'  # compute logits of each frame,
# then average them, then predict for video

VIDEO_PREDS = [FRM_VOTE, FRM_AVG_PROBS, FRM_AVG_LOGITS]
