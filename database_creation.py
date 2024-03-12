import sqlite3
import numpy as np


def insertBLOB(dataset, model, explanable_type,file_path):
    try:
        sqliteConnection = sqlite3.connect('image_database.db')
        cursor = sqliteConnection.cursor()
        print("Connected to SQLite")
        cursor.execute("""
          CREATE TABLE IF NOT EXISTS images (
              dataset TEXT,
              model TEXT,
              explanable_type TEXT,
              file_path TEXT 
          )
          """)
        sqlite_insert_blob_query = """ INSERT INTO images
                                  (dataset, model, explanable_type,file_path) VALUES (?, ?, ?,?)"""


        # Convert data into tuple format
        data_tuple = (dataset, model, explanable_type,file_path)
        cursor.execute(sqlite_insert_blob_query, data_tuple)
        sqliteConnection.commit()
        print("file inserted successfully as a BLOB into a table")
        cursor.close()

    except sqlite3.Error as error:
        print("Failed to insert blob data into sqlite table", error)
    finally:
        if sqliteConnection:
            sqliteConnection.close()
            print("the sqlite connection is closed")
##lr unsw
insertBLOB('UNSW_NB15', 'LR', 'lime', "Results/UNSW nb15/LR/lime_lr.png")
insertBLOB('UNSW_NB15', 'LR', 'Shap', "Results/UNSW nb15/LR/shap_lr_final.png")
insertBLOB('UNSW_NB15', 'LR', 'Feature_imp', "Results/UNSW nb15/LR/feature_imp_lr.png")
insertBLOB('UNSW_NB15', 'LR', 'pdp', "Results/UNSW nb15/LR/pdp_lr.png")
insertBLOB('UNSW_NB15', 'LR', 'weight', "Results/UNSW nb15/LR/weighted_lr.png")
##RF unsw
insertBLOB('UNSW_NB15', 'RF', 'lime', "Results/UNSW nb15/RF/lime_rf.png")
insertBLOB('UNSW_NB15', 'RF', 'Shap', "Results/UNSW nb15/RF/shap_rf_final.png")
insertBLOB('UNSW_NB15', 'RF', 'Feature_imp', "Results/UNSW nb15/RF/feature_imp_rf.png")
insertBLOB('UNSW_NB15', 'RF', 'pdp', "Results/UNSW nb15/RF/pdp final_rf.png")
insertBLOB('UNSW_NB15', 'RF', 'weight', "Results/UNSW nb15/RF/weighted_rf.png")

## LR NSL_KDD
insertBLOB('NSL_KDD', 'LR', 'lime', "Results/NSL kdd/LR/lime_lr.png")
insertBLOB('NSL_KDD', 'LR', 'Shap', "Results/NSL kdd/LR/lr_shap final.png")
insertBLOB('NSL_KDD', 'LR', 'Feature_imp', "Results/NSL kdd/LR/feature_imp_lr.png")
insertBLOB('NSL_KDD', 'LR', 'pdp', "Results/NSL kdd/LR/pdp_lr.png")
insertBLOB('NSL_KDD', 'LR', 'weight', "Results/NSL kdd/LR/weighted_lr.png")
##RF NSL KDD
insertBLOB('NSL_KDD', 'RF', 'lime', "Results/NSL kdd/RF/lime_rf.png")
insertBLOB('NSL_KDD', 'RF', 'Shap', "Results/NSL kdd/RF/rd_shap_final.png")
insertBLOB('NSL_KDD', 'RF', 'Feature_imp', "Results/NSL kdd/RF/feature_imp_rf.png")
insertBLOB('NSL_KDD', 'RF', 'pdp', "Results/NSL kdd/RF/pdp_rf.png")
insertBLOB('NSL_KDD', 'RF', 'weight', "Results/NSL kdd/RF/weighted_rf.png")

