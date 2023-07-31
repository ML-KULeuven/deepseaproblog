import pickle

from examples.DATES.data import create_dataloader


def load_data(DIGITS, CUR):
    """ Creation of all datasets """
    try:
        D = pickle.load(open(f"examples/DATES/data/{DIGITS}_digits" + CUR + "/train_data.pkl", "rb"))
        D_val = pickle.load(open(f"examples/DATES/data/{DIGITS}_digits" + CUR + "/val_data.pkl", "rb"))
        D_test = pickle.load(open(f"examples/DATES/data/{DIGITS}_digits" + CUR + "/test_data.pkl", "rb"))
        D_regressval = pickle.load(open(f"examples/DATES/data/{DIGITS}_digits" + "_regress" + "/val_data.pkl", "rb"))
        # D_classval = pickle.load(open(f"examples/DATES/data/{DIGITS}_digits"+ "_cur" + "/val_data.pkl", "rb"))
        D_regresstest = pickle.load(open(f"examples/DATES/data/{DIGITS}_digits" + "_regress" + "/test_data.pkl", "rb"))
        # D_classtest = pickle.load(open(f"examples/DATES/data/{DIGITS}_digits"+ "_cur" + "/test_data.pkl", "rb"))
        D_cur = pickle.load(open(f"examples/DATES/data/{DIGITS}_digits" + "_regress" + "/train_data.pkl", "rb"))
    except Exception:
        D = create_dataloader("train", batch_size=10, digits=DIGITS, cur=CUR)
        D_val = create_dataloader("val", batch_size=50, digits=DIGITS, cur=CUR)
        D_test = create_dataloader("test", batch_size=50, digits=DIGITS, cur=CUR)
        D_regressval = create_dataloader("val", batch_size=50, digits=DIGITS, cur="_regress")
        # D_classval = create_dataloader("val", batch_size=50, digits=DIGITS, cur="_cur")
        D_regresstest = create_dataloader("test", batch_size=50, digits=DIGITS, cur="_regress")
        # D_classtest = create_dataloader("test", batch_size=50, digits=DIGITS, cur="_cur")
        D_cur = create_dataloader("train", batch_size=10, digits=DIGITS, cur="_regress", data_size=250)
        pickle.dump(D, open(f"examples/DATES/data/{DIGITS}_digits_2023" + CUR + "/train_data.pkl", "wb"))
        pickle.dump(D_val, open(f"examples/DATES/data/{DIGITS}_digits_2023" + CUR + "/val_data.pkl", "wb"))
        pickle.dump(D_test, open(f"examples/DATES/data/{DIGITS}_digits_2023" + CUR + "/test_data.pkl", "wb"))
        pickle.dump(D_regressval,
                    open(f"examples/DATES/data/{DIGITS}_digits_2023" + "_regress" + "/val_data.pkl", "wb"))
        # pickle.dump(D_classval, open(f"examples/DATES/data/{DIGITS}_digits_2023" + "_cur" + "/val_data.pkl", "wb"))
        pickle.dump(D_regresstest,
                    open(f"examples/DATES/data/{DIGITS}_digits_2023" + "_regress" + "/test_data.pkl", "wb"))
        # pickle.dump(D_classtest, open(f"examples/DATES/data/{DIGITS}_digits_2023" + "_cur" + "/test_data.pkl", "wb"))
        pickle.dump(D_cur, open(f"examples/DATES/data/{DIGITS}_digits_2023" + "_regress" + "/train_data.pkl", "wb"))

    return D, D_val, D_test, D_regressval, D_regresstest, D_cur
