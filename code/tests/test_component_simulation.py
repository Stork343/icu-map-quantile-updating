from pathlib import Path
import sys
import unittest

import numpy as np
from numba import set_num_threads

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from adaptive_profile_core import ProfileStay, check_loss, empirical_quantile
from component_learning_selection import fit_selection, predict_selection
from component_simulation_baselines import combined_records, fit_baselines, predict_baselines
from run_adaptive_mixed_simulation import BASE_BETA, HORIZONS, build_stays, simulate_dataset
from verify_component_learning_simulation import score_array


class ComponentSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_num_threads(1)
        cls.data=simulate_dataset("informative_recovery",120,57931)
        cls.fitting=np.arange(72)
        cls.tuning=np.arange(72,96)
        cls.assessment=np.arange(96,120)
        cls.fit_stays=build_stays(cls.data,cls.fitting,BASE_BETA)
        cls.tune_stays=build_stays(cls.data,cls.tuning,BASE_BETA)
        cls.test_stays=build_stays(cls.data,cls.assessment,BASE_BETA)
        cls.records,cls.horizons,cls.identities=combined_records(cls.tune_stays,cls.tuning)
        cls.baseline=fit_baselines(cls.records,cls.horizons,cls.fit_stays)

    def test_original_baseline_contains_116_matched_rules(self):
        for horizon in HORIZONS:
            local=self.baseline["horizons"][str(int(horizon))]
            self.assertEqual(len(local["rules"]),116)
            self.assertEqual(len({row["name"] for row in local["rules"]}),116)
            direct=[row for row in local["rules"] if row["spec"]["basis"]=="level_slope" and not row["calibrated"]]
            self.assertEqual(len(direct),48)
            selected=local["selected"]["original_selected"]
            self.assertAlmostEqual(selected["tuning_loss"],min(row["tuning_loss"] for row in local["rules"]),places=13)

    def test_empirical_quantile_branch_is_exact_on_original_index_values(self):
        from component_simulation_baselines import raw_predictions
        from component_profile_learning import make_profile_batch
        batch=make_profile_batch(self.tune_stays,12.)
        predictions,_=raw_predictions(self.tune_stays,batch,"scalar_0")
        expected=np.repeat([empirical_quantile(stay.index_y) for stay in self.tune_stays],12)
        np.testing.assert_array_equal(predictions,expected)

    def test_stump_uses_fitting_cutpoints_and_one_rule_per_side(self):
        from adaptive_profile_core import index_features
        expected=np.quantile(np.asarray([index_features(stay) for stay in self.fit_stays]),[.25,.5,.75],axis=0)
        for horizon in HORIZONS:
            local=self.baseline["horizons"][str(int(horizon))]
            selector=local["stump"]
            np.testing.assert_allclose(selector["training_quartile_cutpoints"],expected,rtol=0,atol=1e-13)
            names={row["name"] for row in local["rules"]}
            self.assertIn(selector["left_rule_name"],names)
            self.assertIn(selector["right_rule_name"],names)

    def test_frozen_baseline_forecast_uses_no_assessment_outcomes(self):
        test_records,horizons,_=combined_records(self.test_stays,self.assessment)
        changed=[type(stay)(stay.index_residual,stay.index_time,stay.later_y+1000,
            stay.later_time,stay.later_population,index_y=stay.index_y) for stay in test_records]
        first,_,_=predict_baselines(test_records,horizons,self.baseline)
        second,_,_=predict_baselines(changed,horizons,self.baseline)
        for name in first:
            np.testing.assert_array_equal(first[name]["flat_predictions"],second[name]["flat_predictions"])

    def test_shared_inner_selection_retains_stay_fold_separation(self):
        fitted=fit_selection(self.records,self.horizons,self.identities,seed=68319,
            precomputed_constant_losses=self.baseline["constant_loss_matrix"],max_iter=2,deterministic_starts=1)
        self.assertEqual(len(fitted["validation_rows"]),15)
        self.assertEqual(len(fitted["validation_scores"]),5)
        for fold in range(1,4):
            rows=[row for row in fitted["validation_rows"] if row["fold"]==fold]
            self.assertEqual(len({row["training_records"] for row in rows}),1)
            self.assertEqual(len({row["validation_records"] for row in rows}),1)
            self.assertEqual(rows[0]["training_records"]+rows[0]["validation_records"],len(self.records))
        for score in fitted["validation_scores"]:
            contribution=sum(row["validation_contribution"] for row in fitted["validation_rows"] if row["candidate"]==score["candidate"])
            self.assertAlmostEqual(contribution,score["validation_loss"],places=13)

    def test_canonical_published_score_uses_retained_outcomes(self):
        y=np.asarray([[60.,70.,80.,90.]])
        q=np.asarray([[60.,69.,80.,95.]])
        times=np.asarray([[13.,14.,15.,16.]])
        result=score_array(y,q,times,4.)
        np.testing.assert_array_equal(result["p_lt"],[.25])
        np.testing.assert_array_equal(result["p_le"],[.75])
        self.assertAlmostEqual(result["loss"][0],np.mean(check_loss(y-q,.1)),places=15)


if __name__=="__main__": unittest.main()
