import tensorflow as tf

# Define placeholders
a = tf.placeholder(tf.float32, [None, 1])
b = tf.placeholder(tf.float32, [None, 1])
c = tf.placeholder(tf.float32, [None, 1])

# Construct data flow graph
b_neg = tf.mul(b, tf.constant(-1, tf.float32))
b_sq = tf.square(b)
four_a_c = tf.mul(tf.constant(4, tf.float32), tf.mul(a, c))
sqrt = tf.sqrt(tf.sub(b_sq, four_a_c))

nom1 = tf.add(b_neg, sqrt)
nom2 = tf.sub(b_neg, sqrt)

denom = tf.mul(a, tf.constant(2, tf.float32))

x1 = tf.div(nom1, denom)
x2 = tf.div(nom2, denom)


# Define input values
a_in = [[2], [1], [2]]
b_in = [[3], [2], [1]]
c_in = [[-5], [-3], [-1]]

# Create new session and evaluate graph
with tf.Session() as session:
	# Feed data into the graph as python dictionary
	_x1, _x2 = session.run([x1, x2], feed_dict = {a: a_in, b: b_in, c: c_in})
	print(_x1, _x2)
