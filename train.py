import sys
from qpython import qconnection
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, LSTM, Dropout
import numpy as np
from numpy import array
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.constraints import max_norm



# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out, stride=1):
    X, y = list(), list()
    for i in range(0, len(sequences), stride):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def get_time_str(time_index):
    return "%d:%d" %(time_index//60,time_index%60)


def get_time_index(time_str):
    tokens = time_str.split(":")
    return 60*((int)(tokens[0]))+(int)(tokens[1])


#assumes first index filled
def pad_array(array):
    copied_index=0
    for i in range (len(array)):
        if array[i] != 0:
            copied_index=i
        else:
            array[i] = array[copied_index]
    arr = array.reshape((len(array),1))
    return arr


# define model
def create_model(num_hidden_units=100, dropout=0.2, num_layers=2, weight_constraint=3, learn_rate=0.001):

    #print("building model...")
    model = Sequential()
    model.add(LSTM(num_hidden_units, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features), kernel_constraint=max_norm(weight_constraint)))
    model.add(BatchNormalization())


    for i in range(max(0, num_layers-2)):
        model.add(LSTM(num_hidden_units, activation='relu', return_sequences=True, kernel_constraint=max_norm(weight_constraint)))
        model.add(Dropout(dropout))
    model.add(LSTM(num_hidden_units, activation='relu', kernel_constraint=max_norm(weight_constraint)))
    model.add(Dense(n_steps_out))

    #compile model
    optimizer = Adam(lr=learn_rate)
    model.compile(optimizer=optimizer, loss='mse')
    #print(model.summary())

    return model


if __name__ == ('__main__'):
    use_csv = True

    NUM_INDICES_PERDAY = 1291

    # input shape: (samples, time steps, features)

    # multivariate multi-step data preparation

    if use_csv:
        data_file = sys.argv[1]
        data = open(data_file, 'r').read()
    else:
        q = qconnection.QConnection(host='199-DEV-12501', port=5001, pandas=True)
        try:
            q.open()
            data = q('.ml.getData[2019.07.10;`$"FENICS UST ",/:string[2 3 5 7 10 30],\:"Y";08:00 16:00;0D00:01]')
        finally:
            q.close()



    train_out_file = sys.argv[2]
    train_out = open(train_out_file,'w')

    rows = data.splitlines()
    row_index=0
    price = np.zeros(NUM_INDICES_PERDAY)
    bid = np.zeros(NUM_INDICES_PERDAY)
    offer = np.zeros(NUM_INDICES_PERDAY)

    #first fill sequences from data as much as possible
    #replace 100 with len(rows)
    #format with pandas
    for row_index in range(len(rows)):
        dp = rows[row_index].split(",")
        if dp[2] != "FENICS UST 10Y" and dp[2] != "instrument": break
        if '2019-06-17' != dp[0]:
            continue
        else:
            time_index = get_time_index(dp[1])
            #print(time_index)
            price[time_index] = float(dp[3])
            bid[time_index]= float(dp[4])
            offer[time_index]= float(dp[5])

    #PAD and scale data. assumes 0:00 is filled
    scaler = MinMaxScaler()
    price_padded_nonscale = pad_array(price)
    price_padded = scaler.fit_transform(price_padded_nonscale)
    offer_padded = scaler.fit_transform(pad_array(bid))
    bid_padded = scaler.fit_transform(pad_array(offer))

    price_target_padded = (np.roll(price_padded_nonscale, 1))
    scalar_y = scaler.fit(price_target_padded)
    price_target_padded_scaled = scalar_y.transform(price_target_padded)

    sequences = np.hstack((price_padded, offer_padded, bid_padded, price_target_padded_scaled))

    n_steps_in, n_steps_out = 60, 10
    X, y = split_sequences(sequences, n_steps_in, n_steps_out)
    n_features = X.shape[2]

    # define model
    print("building model...")
    NUM_HIDDEN_UNITS = 100
    DROPOUT = 0.2
    NUM_LAYERS = 4
    LEARN_RATE = 0.001
    model = create_model(
        num_hidden_units=NUM_HIDDEN_UNITS,
        dropout=DROPOUT,
        num_layers=NUM_LAYERS,
        learn_rate=LEARN_RATE)


    # fit model
    print("training...")
    NUM_EPOCHS = 20
    BATCH_SIZE = 32
    train_test_split = 0.2
    history = model.fit(X, y, epochs=NUM_EPOCHS, verbose=True, batch_size=BATCH_SIZE, validation_split=train_test_split)
    model.save('my_model_2.0.h5')




    #plot loss during training
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # demonstrate prediction (fix)
    print("predicting ...")
    predicted_prices_scaled = []
    chopped = X[::n_steps_out]

    #inverse scale here
    predicted_prices_scaled = model.predict(chopped, verbose=True).flatten()
    predicted_prices = scalar_y.inverse_transform(predicted_prices_scaled.reshape(-1,1))
    real_prices = np.copy(price_padded_nonscale)

    for i in range(len(predicted_prices)):
        train_out.write("%f,%f\n" % (predicted_prices[i], real_prices[i + n_steps_in+1]))

    # plot predicted and real prices on matplotlib
    x_axis = np.arange(len(predicted_prices)) + 1
    plt.plot(x_axis, predicted_prices, x_axis, real_prices[n_steps_in+1:])
    plt.axvline(x=(1-train_test_split)*len(predicted_prices), color='r')
    plt.title('predicted prices for 10Y UST on FENICS 6/10/19')
    plt.ylabel('lastPx')
    plt.xlabel('timesteps (minutes)')
    plt.legend(['prediction','real'], loc='upper left')
    plt.show()


    # plot residual distribution
    plt.hist(predicted_prices-real_prices[n_steps_in+1:], 50, density=True, facecolor='g', alpha=0.75)
    plt.grid(True)
    plt.show()


