import numpy as np


def evaluate(EM_hat, M, A_hat, A_true, Y, Y_hat):
    # HSI
    norm_y = np.sqrt(np.sum(Y ** 2, 1))
    norm_y_hat = np.sqrt(np.sum(Y_hat ** 2, 1))
    armse_y = np.mean(np.sqrt(np.mean((Y_hat - Y) ** 2, axis=1)))
    asad_y = np.mean(np.arccos(np.sum(Y_hat * Y, 1) / norm_y / norm_y_hat))

    # Endmembers
    p = A_true.shape[0]
    sad_err = [0] * p
    for i in range(p):
        norm_EM_GT = np.sqrt(np.sum(M[:, i] ** 2, 0))
        norm_EM_hat = np.sqrt(np.sum(EM_hat[:, i] ** 2, 0))
        sad_err[i] = np.arccos(np.sum(EM_hat[:, i] * M[:, i].T, 0) / norm_EM_hat / norm_EM_GT)

    asad_em = np.mean(sad_err)
    armse_em = np.mean(np.sqrt(np.mean((M - EM_hat) ** 2, axis=0)))


    # Abundances
    class_rmse = [0] * p
    for i in range(p):
        class_rmse[i] = np.sqrt(np.mean((A_hat[i, :] - A_true[i, :]) ** 2, axis=0))
    armse = np.mean(class_rmse)
    armse_a = np.mean(np.sqrt(np.mean((A_hat - A_true) ** 2, axis=0)))

    return armse_y, asad_y, armse_em, sad_err, asad_em, armse_a, class_rmse, armse
