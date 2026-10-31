from pathlib import Path
import sys
import unittest

import numpy as np
from numba import set_num_threads

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_profile_core import ProfileStay, solve_spd_profile
from component_profile_learning import (
    evaluate_component_map, evaluate_component_penalties, fit_component_map,
    initial_parameters, make_profile_batch, penalty_values, subset_profile_batch,
)


def generated_stays(seed=20261022, n=45):
    rng = np.random.default_rng(seed)
    stays = []
    for i in range(n):
        m = (4, 10, 30)[i % 3]
        times = np.sort(rng.uniform(0, 12, m))
        residual = rng.normal(-2., 7., m)
        if i % 7 == 0:
            residual = np.rint(residual)
        stays.append(ProfileStay(residual, times, rng.normal(70., 8., 6),
            np.linspace(13., 24., 6), np.full(6, 70.)))
    return stays


class ComponentProfileLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_num_threads(2)
        cls.stays = generated_stays()
        cls.batch = make_profile_batch(cls.stays, np.tile([4.,8.,12.],15))

    def test_bounded_features_use_index_information_and_continuous_horizon_basis(self):
        changed = [ProfileStay(stay.index_residual, stay.index_time,
            stay.later_y + 1000., stay.later_time, stay.later_population) for stay in self.stays]
        future_changed = make_profile_batch(changed, self.batch.horizons)
        np.testing.assert_array_equal(self.batch.features, future_changed.features)
        self.assertTrue(np.all(np.abs(self.batch.features) <= 1))
        np.testing.assert_array_equal(self.batch.features[:3,:3], np.eye(3))
        intermediate = make_profile_batch(self.stays[:2], np.asarray([6.,10.]))
        np.testing.assert_array_equal(intermediate.features[:,:3], [[.5,.5,0],[0,.5,.5]])

    def test_exact_diagonal_endpoints_agree_with_independent_clarabel(self):
        rng = np.random.default_rng(593)
        penalties = np.exp(rng.uniform(np.log([.03,.03]),np.log([10.,100.]),(len(self.stays),2)))
        penalties[0], penalties[1] = [.03,.03], [10.,100.]
        fitted = evaluate_component_penalties(self.batch, penalties)
        for index, stay in enumerate(self.stays):
            reference, _ = solve_spd_profile(stay.index_residual, stay._index_design, .1, np.diag(penalties[index]))
            np.testing.assert_allclose(fitted["coefficients"][index], reference, atol=3e-7, rtol=1e-8)
        self.assertTrue(fitted["diagnostics"]["all_certified"])
        self.assertLessEqual(fitted["diagnostics"]["max_projected_kkt_violation"],5e-7)
        self.assertLessEqual(fitted["diagnostics"]["max_stationarity_violation"],1e-10)

    def test_near_parallel_active_observations_have_certified_solution(self):
        time_u = np.asarray([-.9,-.5000001,-.5,-.1,0.])
        design = np.column_stack([np.ones(5),time_u])
        penalties = np.asarray([.031,.2])
        dual = np.asarray([.1,-.2,-.3,.1,.1])
        expected = .5 * (design.T @ dual) / penalties
        residual = design @ expected
        residual[[0,3,4]] += [2.,5.,3.]
        stay = ProfileStay(residual,(time_u+1)*12,np.asarray([60.,75.]),np.asarray([15.,24.]),np.full(2,70.))
        batch = make_profile_batch([stay],12.)
        fitted = evaluate_component_penalties(batch,penalties)
        independent, _ = solve_spd_profile(residual,design,.1,np.diag(penalties))
        np.testing.assert_allclose(fitted["coefficients"][0],independent,atol=1e-6,rtol=1e-7)

    def test_hypergradient_matches_unsmoothed_finite_difference(self):
        params = initial_parameters(np.asarray([3.,10.]))
        params[:,3:] = [[.1,-.2,.1],[-.1,.1,.2]]
        result = evaluate_component_map(self.batch,params)
        finite = np.empty_like(params)
        epsilon = 1e-5
        for row in range(2):
            for column in range(6):
                above,below = params.copy(),params.copy()
                above[row,column] += epsilon
                below[row,column] -= epsilon
                finite[row,column] = (evaluate_component_map(self.batch,above)["objective"]
                    -evaluate_component_map(self.batch,below)["objective"])/(2*epsilon)
        np.testing.assert_allclose(result["gradient"],finite,atol=2e-5,rtol=3e-3)

    def test_rank_deficient_dual_face_has_exact_quantile_optimum(self):
        residual=np.asarray([15.31885273034851,19.014642547973224,20.266544834027655,
            25.702696146132695,18.013853281963506,12.19609909278573,15.975480352092575,
            13.1536509822362,20.038470653719315,10.322520452189806])
        time_u=np.asarray([-1.,-31/48,-.625,-23/48,-19/48,-.375,-17/48,-.1875,-1/6,0.])
        stay=ProfileStay(residual,(time_u+1)*12,np.asarray([75.,80.]),np.asarray([15.,24.]),np.full(2,70.))
        batch=make_profile_batch([stay],12.)
        result=evaluate_component_penalties(batch,np.asarray([.03,.03]))
        independent,certificate=solve_spd_profile(residual,np.column_stack([np.ones(len(time_u)),time_u]),.1,
                                                 np.diag([.03,.03]))
        np.testing.assert_allclose(result["coefficients"][0],independent,atol=2e-7,rtol=1e-8)
        self.assertLess(result["diagnostics"]["max_projected_kkt_violation"],5e-7)
        self.assertLess(result["diagnostics"]["max_duality_gap"],1e-7)

    def test_subset_retains_record_scores_and_accepts_explicit_weights(self):
        indices = np.asarray([1,4,7,10])
        weights = np.asarray([.1,.2,.3,.4])
        subset = subset_profile_batch(self.batch,indices,weights=weights)
        original = evaluate_component_penalties(self.batch,np.asarray([.3,1.]))
        selected = evaluate_component_penalties(subset,np.asarray([.3,1.]))
        np.testing.assert_allclose(original["loss"][indices],selected["loss"],atol=1e-10)
        self.assertAlmostEqual(selected["objective"], weights @ original["loss"][indices],places=11)

    def test_learning_metadata_tracks_every_certified_outer_evaluation(self):
        initial = initial_parameters(np.asarray([.3,.3]))
        initial_loss = evaluate_component_map(self.batch,initial)["objective"]
        fitted = fit_component_map(self.batch,np.asarray([.3,.3]),ridge_eta=.01,max_iter=3,deterministic_starts=1)
        self.assertFalse(fitted["global_optimum_certified"])
        self.assertLessEqual(fitted["objective"],initial_loss+1e-10)
        self.assertEqual(fitted["all_evaluation_diagnostics"]["profiles"],fitted["evaluations"]*self.batch.n_records)
        expected = fitted["empirical_loss"]+.01*np.sum(fitted["params"][:,3:]**2)
        self.assertAlmostEqual(fitted["objective"],expected,places=12)
        self.assertEqual(fitted["optimizer_success"],fitted["starts"][0]["optimizer_success"])

    def test_smooth_penalty_link_preserves_component_ranges(self):
        params = np.asarray([[-12,12,-12,4,-4,4],[12,-12,12,-4,4,-4]],dtype=float)
        penalties,_ = penalty_values(params,self.batch.features)
        self.assertTrue(np.all(penalties >= [.03,.03]))
        self.assertTrue(np.all(penalties <= [10.,100.]))


if __name__ == "__main__":
    unittest.main()
