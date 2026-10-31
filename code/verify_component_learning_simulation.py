from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import t

sys.path.insert(0,str(Path(__file__).resolve().parent))

from adaptive_profile_core import check_loss, empirical_quantile, fit_affine_calibration, index_features
from component_profile_learning import evaluate_component_map, evaluate_component_penalties, make_profile_batch
from expanded_family_core import candidate_specs
from run_adaptive_mixed_simulation import HORIZONS, SETTINGS, build_stays, simulate_dataset
from split_window_analysis_core import profiled_intercept


SEED=20261022
VERSION="continuous_components_20261022_v1"
RIDGES=(0.,.001,.01,.1)
METHODS=("continuous_direct","continuous_affine","selected_direct","selected_affine",
         "constant_direct","constant_affine","vector_affine","original_selected",
         "stump_original","calibrated_q10","population")
CONTRASTS=(("continuous_direct","constant_direct"),("selected_direct","constant_direct"),
           ("continuous_affine","constant_affine"),("selected_affine","constant_affine"),
           ("selected_affine","stump_original"),("selected_affine","original_selected"),
           ("continuous_affine","original_selected"),("selected_affine","calibrated_q10"))
PAIR_COUNT=192
GRID=np.asarray([[row["lambda_level"],row["lambda_slope"]] for row in candidate_specs()
                 if row["basis"]=="level_slope"])


def require(condition,message):
    if not condition:
        raise ValueError(message)


def compare_frames(actual,expected,keys,columns,atol=1e-7):
    require(not actual.duplicated(keys).any() and not expected.duplicated(keys).any(),"Repeated result keys")
    left,right=actual.set_index(keys).sort_index(),expected.set_index(keys).sort_index()
    require(left.index.equals(right.index),"Incomplete result grid")
    np.testing.assert_allclose(left[columns].to_numpy(dtype=float),right[columns].to_numpy(dtype=float),rtol=0,atol=atol)


def weights(horizons):
    result=np.empty(len(horizons))
    for horizon in HORIZONS:
        selected=horizons==horizon
        require(np.any(selected),"Missing forecast horizon")
        result[selected]=1/(3*selected.sum())
    return result


def panel(stays,identities):
    return stays*3,np.repeat(HORIZONS,len(stays)),np.tile(identities,3)


def score_array(y,prediction,times,horizon):
    selected=times<=12+horizon
    counts=selected.sum(axis=1)
    require(np.all(counts==int(horizon)),"Future observation counts differ")
    residual=y-prediction
    return {"loss":np.sum(check_loss(residual,.1)*selected,axis=1)/counts,
            "p_lt":np.sum((y<prediction)*selected,axis=1)/counts,
            "p_le":np.sum((y<=prediction)*selected,axis=1)/counts}


def raw_library(stays,beta):
    batch=make_profile_batch(stays,12.)
    population=np.asarray([stay.later_population for stay in stays])
    times=np.asarray([stay.later_time for stay in stays])
    y=np.asarray([stay.later_y for stay in stays])
    result={"population":population}
    diagnostics=[]
    for spec in candidate_specs()[1:]:
        key=spec["key"]
        if spec["basis"]=="level_slope":
            evaluated=evaluate_component_penalties(batch,np.asarray([spec["lambda_level"],spec["lambda_slope"]]))
            result[key]=evaluated["flat_predictions"].reshape(len(stays),12)
            diagnostics.append(evaluated["diagnostics"])
        elif spec["lambda_level"]==0:
            result[key]=np.asarray([np.full(12,empirical_quantile(stay.index_y)) for stay in stays])
        else:
            offsets=np.asarray([profiled_intercept(stay.index_residual,.1,spec["lambda_level"]) for stay in stays])
            result[key]=population+offsets[:,None]
    for name in ("last","minimum","mean","below65"):
        values=[]
        for stay in stays:
            if name=="last": value=stay.index_y[-1]
            elif name=="minimum": value=np.min(stay.index_y)
            elif name=="mean": value=np.mean(stay.index_y)
            else: value=np.mean(stay.index_y<65)
            values.append(np.full(12,value))
        result[f"summary_{name}"]=np.asarray(values)
    return result,y,times,diagnostics


def verify_baseline_tuning(frozen,fit_stays,tune_stays,beta):
    library,y,times,diagnostics=raw_library(tune_stays,beta)
    specifications={row["key"]:row for row in candidate_specs()}
    expected_names=set(specifications)|{key+"__affine" for key in specifications}
    expected_names|={f"summary_{name}__affine" for name in ("last","minimum","mean","below65")}
    fitting_features=np.asarray([index_features(stay) for stay in fit_stays])
    tuning_features=np.asarray([index_features(stay) for stay in tune_stays])
    cutpoints=np.quantile(fitting_features,[.25,.5,.75],axis=0)
    constant_losses=[]
    for horizon in HORIZONS:
        local=frozen["baseline_horizons"][str(int(horizon))]
        rules=local["rules"]
        require(len(rules)==116 and {row["name"] for row in rules}==expected_names,"Original baseline menu must contain 116 canonical rules")
        columns=[]
        mask=times<=12+horizon
        observation_weight=np.repeat(1/horizon,int(mask.sum()))
        for row in rules:
            key=row["name"].removesuffix("__affine")
            require(row["spec"]["key"]==key,"Baseline key and specification differ")
            prediction=row["intercept"]+row["slope"]*library[key]
            values=score_array(y,prediction,times,horizon)["loss"]
            np.testing.assert_allclose(values.mean(),row["tuning_loss"],rtol=0,atol=1e-7)
            if row["calibrated"]:
                require(row["name"].endswith("__affine"),"Baseline calibration flag differs")
                optimum=fit_affine_calibration(y[mask],library[key][mask],observation_weight)
                attained=float(np.sum(check_loss(y[mask]-prediction[mask],.1)*observation_weight))
                np.testing.assert_allclose(attained,optimum["weighted_loss"],rtol=1e-8,atol=1e-7)
            else:
                require(row["intercept"]==0 and row["slope"]==1,"Direct baseline uses an affine map")
            columns.append(values)
        matrix=np.column_stack(columns)
        by_name={row["name"]:index for index,row in enumerate(rules)}
        for name,chosen in local["selected"].items():
            if name=="original_selected": candidates=np.arange(116)
            elif name in ("constant_direct","vector_affine"):
                candidates=np.asarray([index for index,row in enumerate(rules) if row["spec"]["basis"]=="level_slope"
                    and row["calibrated"]==(name=="vector_affine")])
            elif name=="calibrated_q10": candidates=np.asarray([by_name["scalar_0__affine"]])
            elif name=="population": candidates=np.asarray([by_name["population"]])
            else: raise ValueError("Unexpected selected baseline")
            chosen_index=by_name[chosen["name"]]
            require(chosen_index in candidates,"Selected baseline is outside its family")
            require(matrix[:,chosen_index].mean()<=matrix[:,candidates].mean(axis=0).min()+1e-9,
                    "Baseline was not selected using its tuning loss")
        selector=local["stump"]
        np.testing.assert_allclose(selector["training_quartile_cutpoints"],cutpoints,rtol=0,atol=1e-12)
        global_index=int(np.argmin(matrix.mean(axis=0)))
        best=float(matrix[:,global_index].mean())
        for feature in range(3):
            for threshold in np.unique(cutpoints[:,feature]):
                left=tuning_features[:,feature]<=threshold
                if not left.any() or left.all(): continue
                value=(matrix[left].sum(axis=0).min()+matrix[~left].sum(axis=0).min())/len(tune_stays)
                best=min(best,float(value))
        if selector["feature"]==-1:
            assignments=np.full(len(tune_stays),by_name[selector["left_rule_name"]])
        else:
            feature=int(selector["feature"])
            require(np.any(np.isclose(cutpoints[:,feature],selector["threshold"],rtol=0,atol=1e-12)),"Stump cutpoint did not come from fitting histories")
            assignments=np.where(tuning_features[:,feature]<=selector["threshold"],
                                 by_name[selector["left_rule_name"]],by_name[selector["right_rule_name"]])
        attained=float(matrix[np.arange(len(tune_stays)),assignments].mean())
        np.testing.assert_allclose(attained,selector["tuning_loss"],rtol=0,atol=1e-7)
        require(attained<=best+1e-9,"Stump did not minimize the fixed-cutpoint tuning criterion")
        diagonal=[]
        for spec in candidate_specs():
            if spec["basis"]=="level_slope":
                diagonal.append(matrix[:,by_name[spec["key"]]])
        constant_losses.append(np.column_stack(diagonal))
    return np.vstack(constant_losses),diagnostics


def verify_fit_metadata(fit,n_records,eta):
    params=np.asarray(fit["params"],dtype=float)
    require(params.shape==(2,6) and np.isfinite(params).all(),"Invalid continuous parameters")
    require(np.all(np.abs(params[:,:3])<=12+1e-12) and np.all(np.abs(params[:,3:])<=4+1e-12),"Continuous parameter bounds differ")
    require(fit["global_optimum_certified"] is False and fit["profile_certificates_passed"] is True,"Invalid optimization guarantee")
    require(fit["all_evaluation_diagnostics"]["profiles"]==n_records*fit["evaluations"],"Optimization omitted profile certificate counts")
    require(sum(row["evaluations"] for row in fit["starts"])==fit["evaluations"],"Optimization evaluation counts differ")
    for row in fit["starts"]:
        if row["optimizer_status"]==1:
            require(row["optimizer_success"] is False,"Iteration limit is incorrectly labeled converged")
    require(float(fit["ridge_eta"])==eta,"Incorrect continuous complexity")
    return params


def verify_learning(frozen,tune_stays,tune_indices,constant_losses):
    learned=frozen["learning"]
    records,horizons,identities=panel(tune_stays,tune_indices)
    batch=make_profile_batch(records,horizons,weights(horizons))
    unique=np.unique(identities)
    permutation=np.random.default_rng(frozen["derived_seeds"][3]).permutation(len(unique))
    assignment=np.empty(len(unique),dtype=int)
    assignment[permutation]=np.arange(len(unique))%3
    folds=assignment[np.searchsorted(unique,identities)]
    rows=learned["validation_rows"]
    require(len(rows)==15,"Inner cross validation must contain five candidates in every fold")
    rebuilt=[]
    for fold in range(3):
        train=np.flatnonzero(folds!=fold)
        validation=np.flatnonzero(folds==fold)
        fold_rows=[row for row in rows if row["fold"]==fold+1]
        require(len(fold_rows)==5,"Cross-validation fold is incomplete")
        recorded_initial=np.asarray(fold_rows[0]["initial_penalties"],dtype=float)
        choices=[]
        for horizon_index,horizon in enumerate(HORIZONS):
            selected=train[horizons[train]==horizon]
            matches=np.flatnonzero(np.all(GRID==recorded_initial[horizon_index],axis=1))
            require(len(matches)==1,"Recorded initial penalty is outside the 48 matrix grid")
            chosen_index=int(matches[0])
            criterion=constant_losses[selected].mean(axis=0)
            require(criterion[chosen_index]<=criterion.min()+1e-9,"Inner constant initialization used more than its training records")
            choices.append(chosen_index)
        initial=GRID[choices]
        for row in fold_rows:
            np.testing.assert_allclose(row["initial_penalties"],initial,rtol=0,atol=1e-13)
            require(row["training_records"]==len(train) and row["validation_records"]==len(validation),"CV fold record counts differ")
            if row["candidate"]=="constant":
                losses=np.empty(len(validation))
                for j,horizon in enumerate(HORIZONS):
                    selected=horizons[validation]==horizon
                    losses[selected]=constant_losses[validation[selected],choices[j]]
            else:
                eta=float(row["ridge_eta"])
                require(eta in RIDGES and row["candidate"]==f"continuous_{eta:g}","Unexpected CV candidate")
                params=verify_fit_metadata(row["fit"],len(train),eta)
                train_batch=make_profile_batch([records[i] for i in train],horizons[train],weights(horizons[train]))
                fitted=evaluate_component_map(train_batch,params)
                np.testing.assert_allclose(fitted["objective"],row["fit"]["empirical_loss"],rtol=0,atol=1e-7)
                np.testing.assert_allclose(fitted["objective"]+eta*np.sum(params[:,3:]**2),row["fit"]["objective"],rtol=0,atol=1e-7)
                heldout=make_profile_batch([records[i] for i in validation],horizons[validation],weights(horizons[validation]))
                losses=evaluate_component_map(heldout,params)["loss"]
            local=float(weights(horizons[validation])@losses)
            contribution=float(batch.weights[validation]@losses)
            np.testing.assert_allclose([local,contribution],[row["validation_loss"],row["validation_contribution"]],rtol=0,atol=1e-7)
            rebuilt.append({"candidate":row["candidate"],"contribution":contribution})
    scores=[]
    for name in ["constant"]+[f"continuous_{eta:g}" for eta in RIDGES]:
        scores.append(sum(row["contribution"] for row in rebuilt if row["candidate"]==name))
    np.testing.assert_allclose(scores,[row["validation_loss"] for row in learned["validation_scores"]],rtol=0,atol=1e-7)
    best=int(np.argmin(scores[1:]))
    require(float(learned["best_continuous_eta"])==RIDGES[best],"Complexity was not selected from out-of-fold tuning scores")
    chosen="constant" if scores[0]<=scores[best+1]+1e-12 else "continuous"
    require(learned["selected_kind"]==chosen,"Final branch did not use the out-of-fold criterion")
    indices=[int(index) for index in learned["constant_grid_indices"]]
    require(len(indices)==3 and all(0<=index<48 for index in indices),"Full tuning constant grid index is invalid")
    for horizon,index in zip(HORIZONS,indices):
        criterion=constant_losses[horizons==horizon].mean(axis=0)
        require(criterion[index]<=criterion.min()+1e-9,"Full tuning constant rule does not minimize its 48 matrix criterion")
    np.testing.assert_allclose(GRID[indices],learned["constant_penalties"],rtol=0,atol=1e-13)
    params=verify_fit_metadata(learned["best_continuous"],len(records),RIDGES[best])
    continuous=evaluate_component_map(batch,params)
    np.testing.assert_allclose(continuous["objective"],learned["best_continuous"]["empirical_loss"],rtol=0,atol=1e-7)
    np.testing.assert_allclose(continuous["objective"]+RIDGES[best]*np.sum(params[:,3:]**2),
                               learned["best_continuous"]["objective"],rtol=0,atol=1e-7)
    penalties=np.empty((len(records),2))
    for j,horizon in enumerate(HORIZONS): penalties[horizons==horizon]=GRID[indices[j]]
    constant=evaluate_component_penalties(batch,penalties)
    response=batch.later_residual+batch.later_population
    for name,predictions in (("continuous",continuous),("constant",constant)):
        for horizon in HORIZONS:
            selected=batch.horizons==horizon
            obs=np.repeat(selected,np.diff(batch.later_indptr))
            sizes=np.diff(batch.later_indptr)[selected]
            coefficients=frozen["post_calibrations"][f"{name}_{int(horizon)}"]
            prediction=coefficients["intercept"]+coefficients["slope"]*predictions["flat_predictions"][obs]
            weight=np.repeat(1/sizes,sizes)
            optimum=fit_affine_calibration(response[obs],predictions["flat_predictions"][obs],weight)
            attained=np.sum(check_loss(response[obs]-prediction,.1)*weight)
            np.testing.assert_allclose(attained,optimum["weighted_loss"],rtol=1e-8,atol=1e-7)
    return params,np.asarray(learned["constant_penalties"])


def replay_population(population,indices,beta,frozen,params,constant_penalties):
    stays=build_stays(population,indices,beta)
    records,horizons,_=panel(stays,indices)
    batch=make_profile_batch(records,horizons,weights(horizons))
    continuous=evaluate_component_map(batch,params)
    penalties=np.vstack([np.tile(pair,(len(stays),1)) for pair in constant_penalties])
    constant=evaluate_component_penalties(batch,penalties)
    y=np.asarray([stay.later_y for stay in stays])
    times=np.asarray([stay.later_time for stay in stays])
    population_prediction=np.asarray([stay.later_population for stay in stays])
    raw={"population":population_prediction}
    needed={}
    for local in frozen["baseline_horizons"].values():
        rules={row["name"]:row for row in local["rules"]}
        for row in list(local["selected"].values())+[rules[local["stump"]["left_rule_name"]],rules[local["stump"]["right_rule_name"]]]:
            needed[row["spec"]["key"]]=row["spec"]
    base_batch=make_profile_batch(stays,12.)
    for key,spec in needed.items():
        if key in raw: continue
        if spec["basis"]=="level_slope":
            raw[key]=evaluate_component_penalties(base_batch,np.asarray([spec["lambda_level"],spec["lambda_slope"]]))["flat_predictions"].reshape(len(stays),12)
        elif spec["basis"]=="level":
            if spec["lambda_level"]==0:
                raw[key]=np.asarray([np.full(12,empirical_quantile(stay.index_y)) for stay in stays])
            else:
                offsets=np.asarray([profiled_intercept(stay.index_residual,.1,spec["lambda_level"]) for stay in stays])
                raw[key]=population_prediction+offsets[:,None]
        else:
            name=spec["summary"]
            summaries=[]
            for stay in stays:
                if name=="last": value=stay.index_y[-1]
                elif name=="minimum": value=np.min(stay.index_y)
                elif name=="mean": value=np.mean(stay.index_y)
                else: value=np.mean(stay.index_y<65)
                summaries.append(np.full(12,value))
            raw[key]=np.asarray(summaries)
    outputs={}
    for j,horizon in enumerate(HORIZONS):
        local=frozen["baseline_horizons"][str(int(horizon))]
        rules={row["name"]:row for row in local["rules"]}
        for name,row in local["selected"].items():
            outputs[(horizon,name)]=score_array(y,row["intercept"]+row["slope"]*raw[row["spec"]["key"]],times,horizon)
        selector=local["stump"]
        features=np.asarray([index_features(stay) for stay in stays])
        if selector["feature"]==-1: left=np.ones(len(stays),dtype=bool)
        else: left=features[:,selector["feature"]]<=selector["threshold"]
        prediction=np.empty_like(y)
        for group,name in ((left,selector["left_rule_name"]),(~left,selector["right_rule_name"])):
            row=rules[name]
            prediction[group]=row["intercept"]+row["slope"]*raw[row["spec"]["key"]][group]
        outputs[(horizon,"stump_original")]=score_array(y,prediction,times,horizon)
        for name,evaluated in (("continuous",continuous),("constant",constant)):
            selected=horizons==horizon
            # 最终发表评分统一用原始Y与返回预测比较。
            obs=np.repeat(selected,np.diff(batch.later_indptr))
            counts=np.diff(batch.later_indptr)[selected]
            response=np.concatenate([stay.later_y[stay.later_time<=12+horizon] for stay in stays])
            for form in ("direct","affine"):
                values=evaluated["flat_predictions"][obs].copy()
                if form=="affine":
                    cal=frozen["post_calibrations"][f"{name}_{int(horizon)}"]
                    values=cal["intercept"]+cal["slope"]*values
                starts=np.r_[0,np.cumsum(counts)[:-1]]
                outputs[(horizon,f"{name}_{form}")]={
                    "loss":np.add.reduceat(check_loss(response-values,.1),starts)/counts,
                    "p_lt":np.add.reduceat((response<values).astype(float),starts)/counts,
                    "p_le":np.add.reduceat((response<=values).astype(float),starts)/counts}
        for form in ("direct","affine"):
            outputs[(horizon,f"selected_{form}")]=outputs[(horizon,f"{frozen['learning']['selected_kind']}_{form}")]
    return outputs


def verify_replicate(job):
    output,setting,number=job
    directory=Path(output)/"replicates"/setting/f"rep_{number:04d}"
    frozen=json.loads((directory/"frozen.json").read_text())
    completed=json.loads((directory/"complete.json").read_text())
    require(completed["status"]=="complete" and completed["version"]==VERSION and completed["seed"]==SEED,"Incomplete or changed simulation replicate")
    sequence=np.random.SeedSequence([SEED,SETTINGS.index(setting),number])
    seeds=[int(child.generate_state(1)[0]) for child in sequence.spawn(4)]
    require(frozen["derived_seeds"]==seeds and completed["derived_seeds"]==seeds,"Fresh simulation seeds differ")
    data=simulate_dataset(setting,600,seeds[0])
    order=np.random.default_rng(seeds[1]).permutation(600)
    fitting,tuning,assessment=order[:360],order[360:480],order[480:]
    require(not(set(fitting)&set(tuning) or set(fitting)&set(assessment) or set(tuning)&set(assessment)),"Fitting, tuning and assessment overlap")
    coefficients=pd.read_csv(directory/"population_coefficients.csv")
    beta=coefficients.set_index("term").loc[["intercept","age_z","male","urgent"],"estimate"].to_numpy()
    np.testing.assert_allclose(beta,frozen["population_coefficients"],rtol=0,atol=1e-12)
    require(set(coefficients.solver)=={"br"} and coefficients.dual_stationarity.max()<=1e-6
            and coefficients.relative_objective_gap.max()<=1e-7,"Population quantile fit certificate failed")
    observations=0
    primal=0.
    for index in fitting:
        outcomes=np.r_[data.index_y[index],data.later_y[index]]
        observations+=len(outcomes)
        primal+=float(np.sum(check_loss(outcomes-data.x[index]@beta,.1)))
    require(np.all(coefficients.n_observations==observations),"Population fit observation count does not match fitting records")
    np.testing.assert_allclose(coefficients.primal_loss.to_numpy(),primal,rtol=1e-8,atol=1e-7)
    fitting_stays,tuning_stays=build_stays(data,fitting,beta),build_stays(data,tuning,beta)
    constant_losses,diagnostics=verify_baseline_tuning(frozen,fitting_stays,tuning_stays,beta)
    params,constant_penalties=verify_learning(frozen,tuning_stays,tuning,constant_losses)
    reference=simulate_dataset(setting,2000,seeds[2])
    methods,paired,records=[],[],[]
    for scope,population,indices in (("assessment",data,assessment),("reference",reference,np.arange(2000))):
        outputs=replay_population(population,indices,beta,frozen,params,constant_penalties)
        for horizon in HORIZONS:
            for name in METHODS:
                values=outputs[(horizon,name)]
                loss,lower,upper=[float(values[key].mean()) for key in ("loss","p_lt","p_le")]
                methods.append({"scope":scope,"horizon_hours":horizon,"method":name,"n_stays":len(indices),
                    "mean_loss":loss,"p_lt":lower,"p_le":upper,"bracket_violation":max(lower-.1,.1-upper,0),
                    "within_sample_se":float(values["loss"].std(ddof=1)/np.sqrt(len(indices)))})
                if scope=="assessment":
                    records.append(pd.DataFrame({"horizon_hours":horizon,"method":name,"synthetic_stay_index":indices,**values}))
            for first,second in CONTRASTS:
                difference=outputs[(horizon,first)]["loss"]-outputs[(horizon,second)]["loss"]
                paired.append({"scope":scope,"horizon_hours":horizon,"first":first,"second":second,
                    "n_stays":len(indices),"mean_difference":difference.mean(),
                    "within_sample_se":difference.std(ddof=1)/np.sqrt(len(indices))})
    methods,paired,records=pd.DataFrame(methods),pd.DataFrame(paired),pd.concat(records,ignore_index=True)
    compare_frames(pd.read_csv(directory/"methods.csv"),methods,["scope","horizon_hours","method"],
                   ["n_stays","mean_loss","within_sample_se"])
    compare_frames(pd.read_csv(directory/"methods.csv"),methods,["scope","horizon_hours","method"],
                   ["p_lt","p_le","bracket_violation"],atol=1e-12)
    compare_frames(pd.read_csv(directory/"paired.csv"),paired,["scope","horizon_hours","first","second"],
                   ["n_stays","mean_difference","within_sample_se"])
    saved=pd.read_parquet(directory/"assessment_metrics.parquet")
    compare_frames(saved,records,["horizon_hours","method","synthetic_stay_index"],["loss"])
    compare_frames(saved,records,["horizon_hours","method","synthetic_stay_index"],["p_lt","p_le"],atol=1e-12)
    diagnostic=completed["diagnostics"]
    require(diagnostic["all_certified"] is True and diagnostic["max_projected_kkt_violation"]<=5e-7
            and diagnostic["max_stationarity_violation"]<=1e-10,"A recorded profile certificate failed")
    return {"methods":methods.assign(setting=setting,replicate=number),
            "paired":paired.assign(setting=setting,replicate=number),
            "stay_scores":len(records),"recorded_profile_solves":diagnostic["recorded_profile_solves"]}


def verify(output,workers=1):
    output=Path(output)
    design=json.loads((output/"design.json").read_text())
    require(design["version"]==VERSION and design["master_seed"]==SEED,"Simulation protocol differs")
    require(design["settings"]==list(SETTINGS) and design["comparison_count"]==PAIR_COUNT,"Comparison family differs")
    require(design["protocol"]==Path(__file__).with_name("COMPONENT_LEARNING_PROTOCOL.md").read_text(),"Saved protocol text differs")
    replicates=int(design["replicates_per_setting"])
    require(2<=replicates<=200,"Invalid number of replicates")
    jobs=[(str(output),setting,number) for setting in SETTINGS for number in range(1,replicates+1)]
    results=[]
    if workers==1:
        for count,job in enumerate(jobs,1):
            results.append(verify_replicate(job))
            if count%40==0 or count==len(jobs):
                print(f"Independently verified {count}/{len(jobs)} component replicates",flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for count,result in enumerate(pool.map(verify_replicate,jobs),1):
                results.append(result)
                if count%40==0 or count==len(jobs):
                    print(f"Independently verified {count}/{len(jobs)} component replicates",flush=True)
    methods=pd.concat([result["methods"] for result in results],ignore_index=True)
    pairs=pd.concat([result["paired"] for result in results],ignore_index=True)
    for name,expected,keys,columns in (("methods",methods,["setting","replicate","scope","horizon_hours","method"],
        ["n_stays","mean_loss","within_sample_se"]),("paired",pairs,["setting","replicate","scope","horizon_hours","first","second"],
        ["n_stays","mean_difference","within_sample_se"])):
        compare_frames(pd.read_csv(output/f"{name}_replicates.csv"),expected,keys,columns)
    summaries=[]
    for key,group in methods.groupby(["setting","scope","horizon_hours","method"],sort=False):
        require(len(group)==replicates and set(group.replicate)==set(range(1,replicates+1)),"Method omitted an independent replicate")
        lower,upper=group.p_lt.mean(),group.p_le.mean()
        summaries.append({**dict(zip(("setting","scope","horizon_hours","method"),key)),"replicates":replicates,
            "mean_loss":group.mean_loss.mean(),"mcse":group.mean_loss.std(ddof=1)/np.sqrt(replicates),
            "p_lt":lower,"p_le":upper,"mean_bracket_violation":max(lower-.1,.1-upper,0),
            "mean_replicate_violation":group.bracket_violation.mean()})
    compare_frames(pd.read_csv(output/"method_summary.csv"),pd.DataFrame(summaries),["setting","scope","horizon_hours","method"],
                   ["replicates","mean_loss","mcse","p_lt","p_le","mean_bracket_violation","mean_replicate_violation"])
    summaries=[]
    for key,group in pairs.groupby(["setting","scope","horizon_hours","first","second"],sort=False):
        require(len(group)==replicates and set(group.replicate)==set(range(1,replicates+1)),"Interval omitted an independent replicate")
        mean,mcse=group.mean_difference.mean(),group.mean_difference.std(ddof=1)/np.sqrt(replicates)
        point=t.ppf(.975,replicates-1)*mcse
        simultaneous=t.ppf(1-.05/(2*PAIR_COUNT),replicates-1)*mcse
        summaries.append({**dict(zip(("setting","scope","horizon_hours","first","second"),key)),"replicates":replicates,
            "mean_difference":mean,"mcse":mcse,"pointwise_low":mean-point,"pointwise_high":mean+point,
            "simultaneous_low":mean-simultaneous,"simultaneous_high":mean+simultaneous})
    require(len(summaries)==PAIR_COUNT,"Incomplete interval family")
    compare_frames(pd.read_csv(output/"paired_summary.csv"),pd.DataFrame(summaries),["setting","scope","horizon_hours","first","second"],
                   ["replicates","mean_difference","mcse","pointwise_low","pointwise_high","simultaneous_low","simultaneous_high"])
    return {"status":"passed" if replicates==200 else "partial_design_check","replicates_checked":len(jobs),
        "fresh_master_seed":SEED,"paired_intervals_checked":PAIR_COUNT,
        "assessment_stay_scores_checked":sum(result["stay_scores"] for result in results),
        "recorded_profile_solves":sum(result["recorded_profile_solves"] for result in results),
        "all_baseline_tuning_rules_checked":116*3*len(jobs),"all_inner_validation_predictions_replayed":True,
        "frozen_parameters_retrained":False,"final_scoring":"raw retained Y compared with actual returned prediction",
        "visual_inspection":False}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    parser.add_argument("--workers",type=int,default=1)
    args=parser.parse_args()
    require(args.workers>=1,"Verifier worker count must be positive")
    report=verify(args.output,args.workers)
    (args.output/"independent_verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
