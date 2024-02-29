import numpy as np
import matplotlib.pyplot as plt

# FUNCTIONS -------------------------------------------------------------------

# Calculate score that'll go into the function (ex. Sigmoid)
# Note: Score is our value before going into the function.
def get_score(x1, w1, x2, w2, x3, w3, biasW):
    return x1 * w1 + x2 * w2 + x3 * w3 + biasW


# Step Function
# We want the output to be close to our target.
def func_step(x, threshold):
    if x > threshold:
        return 1
    return 0


# Sign Function
# We want the output to be close to our target.
def func_sign(x):
    if x > 0:
        return 1
    return 0


# Sigmoid Function
# We want the output to be close to our target.
def func_sigmoid(x):
    return 1/(1+np.exp(-x))


# UPDATING WEIGHTS ------------------------------------------------------------

# Calculate Error
def calc_error(target, prediction):
    return 2 * (target - prediction)

# This formula needs to be changed!
# From the slides, it should be:
# Error = square of (Prediction - Target)
# New W = Old Weight + learning_rate * error * input
def update_weight(old_weight, target, prediction, value_in_dataset):
    error = calc_error(target, prediction)
    return old_weight + learning_rate * error * prediction * (1 - prediction) * value_in_dataset
    # Is this only for Sigmoid?
    # --> prediction * (1 - prediction)

# Retreive a random entry from a list
def select_random_entry(list):
    entry_index = np.random.randint(len(list))
    entry = list[entry_index]
    return entry_index, entry

def update_new(w1, w2, w3, biasW, epoch):
    tmp_list = dataset
    tmp_len = len(tmp_list)
    for i in range(tmp_len):
        # START
        # Get a random entry from the dataset
        tmp_index, entry = select_random_entry(tmp_list)

        # Entry Values:
        x1 = entry[0]
        x2 = entry[1]
        x3 = entry[2]


        index_offset = (epoch)*len(dataset)

        epoch_history[i+index_offset][0] = epoch+1
        epoch_history[i+index_offset][1] = x1
        epoch_history[i+index_offset][2] = x2
        epoch_history[i+index_offset][3] = x3
        epoch_history[i+index_offset][4] = round(w1, round_amount)
        epoch_history[i+index_offset][5] = round(w2, round_amount)
        epoch_history[i+index_offset][6] = round(w3, round_amount)
        epoch_history[i+index_offset][7] = 1
        epoch_history[i+index_offset][8] = round(biasW, round_amount)


        # Get score, hopefully it's close to the target!
        score = get_score(x1, w1, x2, w2, x3, w3, biasW)
        epoch_history[i+index_offset][9] = round(score, round_amount)
        # Choose function here
        prediction = func_sigmoid(score)
        epoch_history[i+index_offset][10] = round(prediction, round_amount)


        target = entry[3]
        epoch_history[i+index_offset][11] = target
        epoch_history[i+index_offset][12] = round(calc_error(target, prediction), round_amount)
        epoch_history[i+index_offset][13] = learning_rate

        # Update weights
        w1 = update_weight(w1, target, prediction, entry[0])
        w2 = update_weight(w2, target, prediction, entry[1])
        w3 = update_weight(w3, target, prediction, entry[2])
        biasW = update_weight(biasW, target, prediction, 1)
        #tmp_list.pop(tmp_index)

    return w1, w2, w3, biasW


# ALGORITHM -------------------------------------------------------------------

# Output: Error_history, New Weight1, New Weight2, and BiasWeight.
def train(w1, w2, w3, biasW):

    # Train on each iteration
    for i in range(iterations):

        # Store total error-rate into history (Using current weights)
        if i % display_every == 0:
            tmp_error = 0  # Total error counter
            # Test against all in our dataset
            for j in range(len(dataset)):
                # Choose the jth entry, and retrieve their x1, x2, and output values
                tmp_entry = dataset[j]
                # Entry Values:
                t_x1 = tmp_entry[0]
                t_x2 = tmp_entry[1]
                t_x3 = tmp_entry[2]
                t_target = tmp_entry[3]

                tmp_score = get_score(t_x1, w1, t_x2, w2, t_x3, w3, biasW)
                tmp_prediction = func_sigmoid(tmp_score)

                # Add to total error rate
                tmp_error += np.square(tmp_prediction - t_target)

            # Store in history
            error_history.append(tmp_error)
            w1_history.append(w1)
            w2_history.append(w2)
            w3_history.append(w3)
            biasW_history.append(biasW)

        # Update weights
        w1, w2, w3, biasW = update_new(w1, w2, w3, biasW, i)

    return error_history, w1, w2, w3, biasW


# Run the main training function, but with random weights
def train_random():
    w1      = np.random.randn()
    w2      = np.random.randn()
    w3      = np.random.randn()
    biasW   = np.random.randn()

    # Display Weights
    print()
    print("Random Weights:")
    print("Weight 1:\t", round(w1, round_amount))
    print("Weight 2:\t", round(w2, round_amount))
    print("Weight 3:\t", round(w2, round_amount))
    print("Bias Weight:", round(biasW, round_amount))
    print()

    return train(w1, w2, w3, biasW)

# PRESENTATION ----------------------------------------------------------------


# Display how well the algorithm improved
def display_errors():
    print("Printings errors...")
    for i in range(len(errors)):
        iteration = i*display_every
        print("#" + str(iteration) + ":", "\t", round(errors[i], round_amount))


# Plot how the weights change over time.
def plot_graph():
    x_values = range(int(iterations/display_every))
    plt.plot(x_values, w1_history, label='Weight 1')
    plt.plot(x_values, w2_history, label='Weight 2')
    plt.plot(x_values, w3_history, label='Weight 3')
    plt.plot(x_values, biasW_history, label='Bias Weight')
    plt.plot(x_values, errors, label='Error Rate')

    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Weights over time')
    plt.legend()

    # Set y-axis grid lines to dotted
    plt.grid(axis='y', linestyle='--')

    plt.show()

def print_excel_table(*values):
    print("|", end="")
    for val in values:
        print(f"{str(val):^8}|", end="")
    print("\n" + "-" * (8 * len(values) + len(values) + 1))


def print_values_table(values):
    print("|", end="")
    for val in values:
        print(f"{str(val):<8}|", end="")
    print("\n" + "-" * (8 * len(values) + len(values) + 1))


# INPUT & DATASET -------------------------------------------------------------

# Wolf  = 0
# Dog   = 1

dataset = [
        [4,     3,      1,    0],
        [2,     3,      0.5,  1],
        [5,     4,      1.2,  0],
        [4.5,   4.5,    1.1,  0],
        [1,     2.7,    0.5,  1],
        [1.7,   2.7,    0.7,  1],
        [3.5,   3.5,    1.2,  0],
        [2.1,   2.3,    0.5,  1],
        [2.3,   2.1,    0.7,  1],
        [3.5,   3.2,    1.3,  0]
        ]

# New Input
new_data = [3.2, 3.1, 3]

# TRAINING --------------------------------------------------------------------

# Parameters
learning_rate   = 0.1
iterations      = 300
display_every   = 1
round_amount    = 3

# Store history
error_history = []
w1_history = []
w2_history = []
w3_history = []
biasW_history = []

def create_empty_table(x, y):
    table = [[' ' for _ in range(y)] for _ in range(x)]
    return table
epoch_history = create_empty_table(iterations*len(dataset), 14)

# Train
#errors, final_w1, final_w2, final_bias = train_random()
errors, final_w1, final_w2, final_w3, final_bias = train(0.2, 0.1, 0.6, 0.1)

# Display Weights
print_excel_table("Epoch", "X1", "X2", "X3", "W1", "W2", "W3", "B", "BW", "Sum", "Function", "Target", "Error", "LR")
for i in range(len(epoch_history)):
    print_values_table(epoch_history[i])


print()
print( "Final Weights:")
print( "Weight 1:\t", round(final_w1, round_amount) )
print( "Weight 2:\t", round(final_w2, round_amount) )
print( "Weight 3:\t", round(final_w3, round_amount) )
print( "Bias Weight:", round(final_bias, round_amount) )
print()

# Display improvements
#display_errors()

# TESTING ---------------------------------------------------------------------

# Test new input with our trained weights
new_calc = get_score(new_data[0], final_w1, new_data[1], final_w2, new_data[2], final_w3, final_bias)
# Run through our function
new_pred = func_sigmoid(new_calc)


# DISPLAY ---------------------------------------------------------------------

print()
print("New input:", new_data[0], "and", new_data[1], "and", new_data[2])
print("NN Prediction:", round(new_pred, round_amount))
print()

tmp_input = input("Press Enter to view the plotted graph...")
if tmp_input == '':
    plot_graph()

