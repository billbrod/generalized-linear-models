import pytest
import jax
import jax.numpy as jnp
import numpy as np
from nemos.pytrees import FeaturePytree
import nemos as nmo


class TestFeaturePytree:

    def test_key_error_init(self):
        """Ensure TypeError is raised if non-string keys are used during initialization."""
        with pytest.raises(TypeError, match="keywords must be strings"):
            FeaturePytree(**{1: np.random.rand(100)})

    def test_key_error_set(self):
        """Validate that setting a non-string key results in a ValueError."""
        tree = FeaturePytree(**{'test': np.random.rand(100)})
        with pytest.raises(ValueError, match="Keys must be strings"):
            tree[1] = np.random.rand(100)

    def test_num_timepoints_error_init(self):
        """Check for ValueError if arrays with different time points are initialized."""
        with pytest.raises(ValueError, match="All arrays must have same number of time points"):
            FeaturePytree(test1=np.random.rand(100, 1),
                          test2=np.random.rand(50, 1))

    def test_num_timepoints_error_set(self):
        """Ensure setting items with different number of time points raises ValueError."""
        tree = FeaturePytree(test1=np.random.rand(100, 1))
        with pytest.raises(ValueError, match="All arrays must have same number of time points"):
            tree['test2'] = np.random.rand(50, 1)

    def test_array_error_init(self):
        """Validate that initializing with non-array values raises a ValueError."""
        with pytest.raises(ValueError, match="All values must be arrays"):
            FeaturePytree(test1='hi')

    def test_array_error_set(self):
        """Ensure setting non-array values results in a ValueError."""
        tree = FeaturePytree(test1=np.random.rand(100, 1))
        with pytest.raises(ValueError, match="All values must be arrays"):
            tree['test2'] = 'hi'

    def test_diff_shapes(self):
        """Test handling of arrays with different shapes but the same length."""
        tree = FeaturePytree(test=np.random.rand(100))
        for dim in [1, 2, 3, 4]:
            tree[f'test{dim}'] = np.random.rand(100, dim)
        assert len(tree) == 100

    def test_diff_dims(self):
        """Check for correct handling of arrays with different dimensions but the same length."""
        tree = FeaturePytree(test=np.random.rand(100))
        for ndim in [1, 2, 3, 4]:
            tree[f'test{ndim}'] = np.random.rand(100, *[1]*ndim)
        assert len(tree) == 100

    def test_treemap(self):
        """Test the application of jax.tree_map function on FeaturePytree."""
        tree = FeaturePytree(test=np.random.rand(100, 1),
                             test2=np.random.rand(100, 2))
        mapped = jax.tree_map(lambda x: jnp.mean(x, axis=-1), tree)
        assert len(tree) == len(mapped)
        assert list(tree.keys()) == list(mapped.keys())
        assert isinstance(mapped, FeaturePytree)

    def test_treemap_npts(self):
        """Check if jax.tree_map correctly modifies the number of points in FeaturePytree."""
        tree = FeaturePytree(test=np.random.rand(100, 1),
                             test2=np.random.rand(100, 2))
        mapped = jax.tree_map(lambda x: x[::10], tree)
        assert len(mapped) == 10
        assert list(tree.keys()) == list(mapped.keys())

    def test_treemap_to_dict(self):
        """Ensure jax.tree_map transforms FeaturePytree to a dictionary with mean values."""
        tree = FeaturePytree(test=np.random.rand(100,),
                             test2=np.random.rand(100, 2))
        with pytest.warns(UserWarning, match=r"Output is not a FeaturePytree \(e\.g\.\, because at"):
            mapped = jax.tree_map(jnp.mean, tree)
        assert isinstance(mapped, dict)
        assert list(tree.keys()) == list(mapped.keys())

    def test_get_key(self):
        """Test retrieving an item with its key and handling of non-existent keys."""
        test = np.random.rand(100, 1)
        tree = FeaturePytree(test=test)
        np.testing.assert_equal(tree['test'], test)
        with pytest.raises(KeyError):
            tree['hi']

    def test_get_slice(self):
        """Check slicing functionality of FeaturePytree."""
        tree = FeaturePytree(test=np.random.rand(100, 1),
                             test2=np.random.rand(100, 2))
        assert len(tree[:10]) == 10
        assert list(tree.keys()) == list(tree[:10].keys())
        assert all(tree[:10]["test"] == tree["test"][:10])

    def test_glm(self):
        """Validate the Generalized Linear Model (GLM) implementation with FeaturePytree."""
        w_true = FeaturePytree(test=np.random.rand(1, 3),
                               test2=np.random.rand(1, 2))
        X = FeaturePytree(test=np.random.rand(100, 1, 3),
                          test2=np.random.rand(100, 1, 2))
        rate = nmo.utils.pytree_map_and_reduce(lambda w, x: jnp.einsum("ik,tik->ti", w, x),
                                               sum, w_true, X)
        spikes = np.random.poisson(rate)
        model = nmo.glm.GLM()
        model.fit(X, spikes)
        assert list(model.coef_.keys()) == list(X.keys())
        for k in model.coef_.keys():
            assert model.coef_[k].shape == X[k].shape[1:]