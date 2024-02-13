import numpy as np
import matplotlib.pyplot as plt

# FORMULAS --------------------------------------------------------------------

# Calculate score that'll go into the function (ex. Sigmoid)
# Note: Score is our value before going into the function.
def get_score(x1, w1, x2, w2, biasW):
    return x1 * w1 + x2 * w2 + biasW

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

# This formula needs to be changed!
# From the slides, it should be:
# Error = square of (Prediction - Target)
# New W = Old Weight + learning_rate * error * input
def update_weight(old_weight, target, prediction, value_in_dataset):
    error = 2 * (target - prediction)
    return old_weight + learning_rate * error * prediction * (1 - prediction) * value_in_dataset
    # Is this only for Sigmoid?
    # --> prediction * (1 - prediction)


# ALGORITHM -------------------------------------------------------------------

# Output: Error_history, New Weight1, New Weight2, and BiasWeight.
def train(w1, w2, biasW):
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
                t_target = tmp_entry[2]

                tmp_score = get_score(t_x1, w1, t_x2, w2, biasW)
                tmp_prediction = func_sigmoid(tmp_score)

                # Add to total error rate
                tmp_error += np.square(tmp_prediction - t_target)

            #Store in history
            error_history.append(tmp_error)
            w1_history.append(w1)
            w2_history.append(w2)
            biasW_history.append(biasW)

        # START
        # Get a random entry from the dataset
        entry_index = np.random.randint(len(dataset))
        entry = dataset[entry_index]
        # Entry Values:
        x1 = entry[0]
        x2 = entry[1]
        target = entry[2]

        # Get score, hopefully it's close to the target!
        score = get_score(x1, w1, x2, w2, biasW)
        # Choose function here
        prediction = func_sigmoid(score)

        # Update weights
        w1 = update_weight(w1, target, prediction, entry[0])
        w2 = update_weight(w2, target, prediction, entry[1])
        biasW = update_weight(biasW, target, prediction, 1)

    return error_history, w1, w2, biasW


# Run the main training function, but with random weights
def train_random():
    w1      = np.random.randn()
    w2      = np.random.randn()
    biasW   = np.random.randn()

    # Display Weights
    print()
    print("Random Weights:")
    print("Weight 1:\t", round(w1, round_amount))
    print("Weight 2:\t", round(w2, round_amount))
    print("Bias Weight:", round(biasW, round_amount))
    print()

    return train(w1, w2, biasW)

# PRESENTATION ----------------------------------------------------------------

# Display how well the algorithm improved
def display_errors():
    print("Printings errors...")
    for i in range(len(errors)):
        iteration = i*display_every
        print("#" + str(iteration) + ":", "\t", round(errors[i], round_amount))


#Plot how the weights change over time.
def plot_graph():
    x_values = range(int(iterations/display_every))
    plt.plot(x_values, w1_history, label='Weight 1')
    plt.plot(x_values, w2_history, label='Weight 2')
    plt.plot(x_values, biasW_history, label='Bias Weight')
    plt.plot(x_values, errors, label='Error Rate')

    plt.xlabel('Iterations')
    plt.ylabel('Score')
    plt.title('Weights over time')
    plt.legend()

    # Set y-axis grid lines to dotted
    plt.grid(axis='y', linestyle='--')

    plt.show()

# INPUT & DATASET -------------------------------------------------------------

dataset=[
        [3,     1.5,    1],
        [2,     1,      0],
        [4,     1.5,    1],
        [3,     1,      0],
        [3.5,   0.5,    1],
        [2,     0.5,    0],
        [5.5,   1,      1],
        [1,     1,      0]
        ]

# The third column:
# 0 = Blue
# 1 = Red

# New Input
new_data= [4.5, 1]

# TRAINING --------------------------------------------------------------------

# Parameters
learning_rate   = 0.3
iterations      = 5000
display_every   = 5
round_amount    = 3

# Store history
error_history = []
w1_history = []
w2_history = []
biasW_history = []

# Train
errors, final_w1, final_w2, final_bias = train_random()

# Display Weights
print()
print( "Final Weights:")
print( "Weight 1:\t", round(final_w1, round_amount) )
print( "Weight 2:\t", round(final_w2, round_amount) )
print( "Bias Weight:", round(final_bias, round_amount) )
print()

# Display improvements
#display_errors()

# TESTING ---------------------------------------------------------------------

# Test new input with our trained weights
new_calc = get_score(new_data[0], final_w1, new_data[1], final_w2, final_bias)
# Run through our function
new_pred = func_sigmoid(new_calc)


# DISPLAY ---------------------------------------------------------------------

print()
print("New input:", new_data[0], "and", new_data[1])
print("NN Prediction:", round(new_pred, round_amount))
print()

tmp_input = input("Press Enter to view the plotted graph...")
if tmp_input == '':
    plot_graph()