import torch.distributions as dist
from torch import ones

from timeit import timeit
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


n = 3000
d = 3000

# p = 0.3 * ones(n, d).cuda()

print(timeit('ones(n, d)',
      'from __main__ import ones, n, d', number=10))

# x = torch.rand(n, d).to(device)

# b = Binomial(1, 0.3)
# b.sample(torch.Size([n, d]))

# dist = tfp.distributions.Binomial(total_count=1, probs=0.3)
# counts = [1]
# dist.prob(counts)

# binomial_samples = tf.random.stateless_binomial(
#     shape=[1,n,d,1], seed=[123, 456], counts=counts, probs=probs)
# sample = tf.squeeze(binomial_samples)
# sample = sample.numpy()

# print(timeit('tf.random.stateless_binomial(shape=[1,n,d,1], seed=[123, 456], counts=counts, probs=probs)', 'from __main__ import tf, n, d, counts, probs', number=1000))

# shape = [3, 4, 3, 4, 2]
# # Sample shape will be [3, 4, 3, 4, 2]
# binomial_samples = tf.random.stateless_binomial(
#     shape=shape, seed=[123, 456], counts=counts, probs=probs)

# r = binom.rvs(1, p, size=[3000,3000])
# print(timeit('binom.rvs(1, p, size=[3000,3000])', 'from __main__ import binom, p', number=20))

# print(timeit('b.sample(torch.Size([n, d]))', 'from __main__ import b, torch, n, d', number=10))

# print(timeit('Binomial(1, p).sample()', 'from __main__ import Binomial, p', number=10))
# counts = torch.randint(10, 1000, [1000, 1000])
# p = 0.5 * torch.ones(1000, 1000)

# print(timeit('dist.binomial.Binomial(total_count=1, probs=p).sample()',
#       'from __main__ import dist, p', number=10))

# print(timeit('np.random.binomial(1, p, x.shape)', 'from __main__ import np,p,x', number=10))
