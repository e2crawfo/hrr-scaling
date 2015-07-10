import numpy as np
import hrr


def default_noise(vec):
    vector = np.random.rand(len(vec)) - 0.5
    vector = vector / np.linalg.norm(vector)
    return vector


def flip_noise(vec):
    return -vec


def ortho_vector(input_vec, normalize=False):
    """
    Returns a random vector orthogonal to the input vector.

    Args:
    input_vec -- Returned vector will be orthogonal to input_vec
    normalize -- whether to normalize the vector before returning
    """

    normed_input = input_vec / np.linalg.norm(input_vec)
    dim = len(input_vec)
    ortho = hrr.HRR(dim).v
    proj = np.dot(normed_input, ortho) * normed_input
    ortho -= proj
    assert np.allclose([np.dot(ortho, input_vec)], [0])
    if normalize:
        ortho = ortho / np.linalg.norm(ortho)
    return ortho


def sample_cone(input_vec, N=1, angle=None, dot=None):
    if angle is None and dot is None:
        raise ValueError("Have to supply at least one of {angle, dot}")

    if angle is not None:
        dot = np.cos(angle)

    for n in range(N):
        o = ortho_vector(input_vec, normalize=True)

        normed_input = input_vec / np.linalg.norm(input_vec)

        sample = normed_input * dot + o * np.sqrt(1 - dot ** 2)
        sample = np.reshape(sample, (1, sample.size))

        if n == 0:
            samples = sample
        else:
            samples = np.concatenate((samples, sample), axis=0)

    return samples


def output(trial_length, main, main_vector, alternate, noise_func=None):
    tick = 0
    vector = main_vector
    main_hrr = hrr.HRR(data=main_vector)

    noise_func = noise_func if noise_func is not None else default_noise

    while True:
        if tick == trial_length:
            tick = 0
            if main:
                vector = main_vector
            else:
                vector = noise_func(main_vector)
                u = hrr.HRR(data=vector)
                similarity = u.compare(main_hrr)
                print "Sim:", similarity

            if alternate:
                main = not main

        tick += 1

        yield vector


def interpolator(end_time, start_vec, end_vec, time_func=lambda x: x):
    tick = 0
    while True:
        t = time_func(tick)
        t = min(t, end_time)

        vector = (end_time - t) * start_vec + t * end_vec
        vector = vector / np.linalg.norm(vector)
        tick += 1

        yield vector


def make_f(generators, times):
    last_time = [0.0]

    def f(t):
        if len(generators) > 1 and t > times[0] + last_time[0]:
            generators.pop(0)
            last_time[0] += times.pop(0)
        return generators[0].next()
    return f


def make_hrr_from_string(D, expression, names={},
                         normalize=False, verbose=False,
                         as_hrr=False):
    """
    Accepts a string specifying an hrr expression and returns a function that
    produces random instances of HRR vectors of the given form.

    expression is assumed to specify superpositions of convolutions of HRR
    vectors.  The convolved vectors can also be inverted.

    Special subsstrings:
    '?' : Each instance of '?' is replaced by a random HRR vector.
          Vector is generated anew for each call of the returned function.
    '?u' : Same as '?', but only unitary vectors are used

    Vectors filling '?' and '?u' are generated anew for each call of the
    returned function.

    Can fix values assigned to names in the supplied expression using the names
    dictionary. Otherwise random vectors will be generated anew each time the
    returned function is called, just as is done for '?'.

    Some examples of valid strings:

    '? * a + ? * ?u'
    'a * b * c + au * au * d'
    '? * a * c'
    'b * d * b + c * c * c * ~c + d * ~d'

    Args:

    D -- dimension of all vectors.

    expression -- string giving the form of the HRR expression to extract from.

    names -- dictionary mapping strings to HRRs. Can be used to specify values
             for variables that occur in expression. Any variable name that
             does not occur in names will be assigned a random HRR.

    normalize -- whether to normalize the HRR expression before deconvolving.
    """

    original = expression
    expression, unitary_names, temp_names = replace_wildcards(expression)
    print expression

    unitary_names += [u for u in names if u[-1:] == "u"]

    if verbose:
        print 'Evaluated expression: ', expression

    def hrr_from_string():
        """
        Uses expression, names, query_vectors from wrapper function

        Args:

        """

        vocab = hrr.Vocabulary(D, unitary=unitary_names)

        for n, v in names.iteritems():
            vocab.add(n, v)

        try:
            h = eval(expression, {}, vocab)
        except Exception as e:
            print 'Error evaluating HRR string ' + original
            raise e

        if normalize:
            h.normalize()

        if not as_hrr:
            h = h.v

        return h

    return hrr_from_string


def make_hrr_noise_from_string(D, expression, names={},
                               normalize=False, verbose=False):
    """
    Returns a function that takes an input vector and returns a version of that
    vector corrupted by noise. The noise is generated by first constructing an
    HRR vector from the string expression which includes in the input vector,
    and then performing the appropriate operations to extract the input vectors
    from that expression.

    expression is assumed to specify superpositions of convolutions of HRR
    vectors.  The convolved vectors can also be inverted.

    Special subsstrings:
    '!' : Must include exactly one instance. A placeholder for input vector.
    '?' : Each instance of '?' is replaced by a random HRR vector.
    '?u' : Each instance of '?u' is replaced by a random unitary HRR vector.

    Vectors filling '?' and '?u' are generated anew for each call of the
    returned function.

    Some examples of valid strings:

    '? * ! + ? * ?'
    'a * b * c + a * a * !'
    '? * a * !'
    'b * ! * b + c * c * c * ~c + d * ~d'

    Args:

    D -- dimension of all vectors.

    expression -- string giving the form of the HRR expression to extract from.

    names -- dictionary mapping strings to HRRs. Can be used to specify values
             for variables that occur in expression. Any variable name that
             does not occur in names will be assigned a random HRR.

    normalize -- whether to normalize the HRR expression before deconvolving.
    """

    original = expression

    placeholder = 'p0'
    expression = expression.replace('!', placeholder)

    expression, unitary_names, temp_names = replace_wildcards(expression)

    query_vectors = find_query_vectors(expression, placeholder)

    unitary_names += [u for u in names if u[-1:] == "u"]

    if verbose:
        print 'Evaluated expression: ', expression
        extraction_expression = 'h * ~(' + ' * '.join(query_vectors) + ')'
        print 'Extraction expression: ', extraction_expression

    def hrr_noise_from_string(input_vec):
        """
        Uses expression, names, query_vectors from wrapper function

        Args:

        input_vec -- the vector to add noise to. Can be an HRR vector or a
                     numpy ndarry.  Returns a noisy vector of the same type
                     as input_vec.

        """

        use_ndarray = type(input_vec) == np.ndarray
        if use_ndarray:
            input_vec = hrr.HRR(data=input_vec)

        vocab = hrr.Vocabulary(D, unitary=unitary_names)

        for n, v in names.iteritems():
            vocab.add(n, v)

        vocab.add(placeholder, input_vec)

        try:
            h = eval(expression, {}, vocab)
        except Exception as e:
            print 'Error evaluating HRR string ' + original
            raise e

        if normalize:
            h.normalize()

        vocab.add('h', h)

        noisy = eval('h * ~(' + ' * '.join(query_vectors) + ')', {}, vocab)

        if use_ndarray:
            noisy = noisy.v

        return noisy

    return hrr_noise_from_string


def find_query_vectors(expression, placeholder):
    """
    Find the names of the vectors in expression which, when used as a query
    vector, will extract the vector at the position of the placeholder,
    assuming expression is a superposition of convolutions

    Args:

    expression -- string specifiying an HRR expression as
                  superposition of convolutions

    placeholder -- string specifiying the position of the
                   thing we're going to extract

    """
    terms = expression.split('+')

    query_vectors = None
    for term in terms:
        if placeholder in term:
            query_vectors = term.split('*')
            query_vectors = [qv.strip() for qv in query_vectors]
            query_vectors.remove(placeholder)
            break

    if query_vectors is None:
        message = 'HRR string must contain the placeholder: ' + placeholder
        raise ValueError(message)

    return query_vectors


def replace_wildcards(expression):
    """
    Modifies expression to that '?' and '?u' are replaced with unique names.
    Returns the generated names as lists.
    """
    num_unitary_wildcards = expression.count('?u')
    expression = expression.replace('?u', '%s')
    unitary_names = ['u'+str(i) for i in range(num_unitary_wildcards)]
    expression = expression % tuple(unitary_names)

    num_wildcards = expression.count('?')
    expression = expression.replace('?', '%s')
    temp_names = ['h'+str(i) for i in range(num_wildcards)]
    expression = expression % tuple(temp_names)

    return expression, unitary_names, temp_names

if __name__ == "__main__":
    D = 512
    i = hrr.HRR(D)
    f = make_hrr_noise_from_string(D, '?u*?u*?*!*?u')
    x = f(i)
    print i.compare(x)

    i = hrr.HRR(D)
    noise_string = '?*? *? + ? * ? * ? + ? * ? * ! + ?*?'
    f = make_hrr_noise_from_string(D, noise_string, normalize=False)
    x = f(i)
    print i.compare(x)

    i = hrr.HRR(D)
    noise_string = '? * ! + ? * ?'
    f = make_hrr_noise_from_string(D, noise_string, normalize=False)
    x = f(i)
    print i.compare(x)

    i = hrr.HRR(D)
    noise_string = 'a * b * c + a * a * !'
    f = make_hrr_noise_from_string(D, noise_string, normalize=False)
    x = f(i)
    print i.compare(x)

    i = hrr.HRR(D)
    noise_string = '? * a * !'
    f = make_hrr_noise_from_string(D, noise_string, normalize=False)
    x = f(i)
    print i.compare(x)

    i = hrr.HRR(D)
    noise_string = 'b * ! * b + c * c * c * ~c + d * ~d'
    f = make_hrr_noise_from_string(D, noise_string, normalize=False)
    x = f(i)
    print i.compare(x)
