#!/usr/bin/python

# Copyright 2015 Rosanna Milner, Thomas Hain @ SPandh, University of Sheffield
#
#    http://mini.dcs.shef.ac.uk/resources/sw/dia_segmentfmeasure/
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import commands
import itertools


########################################
# INITIAL SETUP
########################################


def initial_setup(args):
    """ This function runs initial setup for the input files.

    It checks wether the input RTTM files are lists or not and reads in the SPEAKER lines, reads scoring times from a UEM file if selected, creates folder for saving output if selected and fixes the collar to be +/-0.005 (1 frame) for the triangular and Gaussian distrbutions when collar is 0.0.
    """
    # check input file formats match
    if args.list:
        reflist = commands.getoutput("cat %s" % args.ref).split("\n")
        hyplist = commands.getoutput("cat %s" % args.hyp).split("\n")
    else:
        reflist = [args.ref]
        hyplist = [args.hyp]
    # find UEM times for scoring
    if args.uem:
        uemlist = [args.uem]
        uem_times = read_uem(uemlist)
    else:
        uem_times = {}
    # read in lines from files
    reflines = {}
    for ref in reflist:
        reflines = read_rttm(ref, reflines)
    hyplines = {}
    for hyp in hyplist:
        hyplines = read_rttm(hyp, hyplines)
    # if saving create folder
    if args.save:
        print "SAVING TO FOLDER:\t%s" % args.folder
        commands.getoutput("mkdir -p %s" % args.folder)
    # check distribution
    if args.collar == 0 and args.distribution != "uniform":
        print "CANNOT RUN GAUSS OR TRI DISTRIBUTION WITH 0 COLLAR, USING FRAME SIZE (0.005 s)"
        args.collar = 0.005
    return [reflines, hyplines, uem_times, args]


def read_uem(uemlist):
    """Reads in UEM file.

    Reads the provided UEM file, checks whether it is a list of files or not and reads in start and end scoring times.
    """
    uem_times = {}
    for uem in uemlist:
        with open(uem) as f1:
            for line in f1:
                x = line.split()
                # check if uem is actually a list of uems
                if len(x) == 1 and ";;" not in line:
                    with open(x[0]) as f2:
                        for line in f2:
                            y = line.split()
                            fname, start, final = y[0], float(y[2]), float(y[3])
                            if fname not in uem_times:
                                uem_times[fname] = [start, final]
                # read single uem file
                elif ";;" not in line:
                    fname, start, final = x[0], float(x[2]), float(x[3])
                    if fname not in uem_times:
                        uem_times[fname] = [start, final]
    return uem_times


def read_rttm(rttm, lines):
    """Reads in SPEAKER lines from RTTM file."""
    with open(rttm) as f:
        for line in f:
            if "SPEAKER" in line:
                fname = line.split()[1]
                if fname not in lines:
                    lines[fname] = []
                lines[fname].append(line)
    return lines


########################################
# READING SEGMENTS
########################################


def organising_segments(name, reflines_file, hyplines_file, uem_times, args):
    """Organises the segments found in the RTTM files.

    It extracts the timings, file name and speaker name from each SPEAKER line in RTTM and saves to a list, restricts the segments to the specified UEM times if selected, merges overlapping segments and gives a single speaker label if SAD selected, smooths the reference segments given a time gap, and finally finds the left the right type of each boundary.
    """
    # read segments
    [refsegments, refspkrs] = read_segments(reflines_file)
    [hypsegments, hypspkrs] = read_segments(hyplines_file)
    # for UEM, find segments within specified scoring times
    if args.uem:
        if name in uem_times:
            refsegments = apply_uem_time(refsegments, uem_times[name])
            hypsegments = apply_uem_time(hypsegments, uem_times[name])
            print "EVAL TIME:\t%.2f TO %.2f" % (uem_times[name][0], uem_times[name][1])
        else:
            print "EVAL TIME:\tCOMPLETE AUDIO"
    else:
        print "EVAL TIME:\tCOMPLETE AUDIO"
    # for speech activity detection, merge segments and label all as SPEECH
    if args.sad:
        refsegments = sad_merge_segments(refsegments)
        hypsegments = sad_merge_segments(hypsegments)
    print "REF SEGMENTS:\t%d" % len(refsegments)
    print "HYP SEGMENTS:\t%d" % len(hypsegments)
    # smooth reference segments for leniency
    if args.gap > 0.0:
        refsegments = smooth_segments(refsegments, args.gap)
        print "REF SEGMENTS (SMOOTHED GAP %.2f):\t%d" % (args.gap, len(refsegments))
    # find boundary labels
    refsegments = find_adjacent_segment_type(refsegments, args.gap)
    hypsegments = find_adjacent_segment_type(hypsegments, args.gap)
    return [refsegments, refspkrs, hypsegments, hypspkrs, args]


def read_segments(lines):
    """Reads beginning time, duration time, speaker name into a list."""
    segments, spkrs = [], {}
    for line in lines:
        x = line.split()
        beg, dur, spkr = float(x[3]), float(x[4]), x[7]
        end = beg + dur
        if spkr not in spkrs:
            spkrs[spkr] = "unknown"
        segments.append([beg, dur, end, spkr])
    segments = sorted(segments, key=lambda s: s[0])
    return [segments, spkrs]


def apply_uem_time(segments, times):
    """Using UEM start and final time, removes segments before and after, and splits if segment spans either start or final time."""
    new_segments, start, final = [], times[0], times[1]
    for [beg, dur, end, spkr] in segments:
        if start < beg and end < final:
            new_segments.append([beg, dur, end, spkr])
        elif beg < start < end and end < final:
            new_segments.append([start, end - start, end, spkr])
        elif start < beg and beg < final < end:
            new_segments.append([beg, final - beg, final, spkr])
        elif beg < start < end and beg < final < end:
            new_segments.append([start, final - start, final, spkr])
    return new_segments


def sad_merge_segments(segments):
    """For SAD, segments given a single speaker label (SPEECH) and overlapping segments are merged."""
    prev_beg, prev_end = 0.0, 0.0
    smoothed_segments = []
    for seg in segments:
        beg, end = seg[0], seg[2]
        if beg > prev_beg and end < prev_end:
            continue
        elif end > prev_end:
            if beg > prev_end:
                smooth = [prev_beg, (prev_end - prev_beg), prev_end, "speech"]
                smoothed_segments.append(smooth)
                prev_beg = beg
                prev_end = end
            else:
                prev_end = end
    smooth = [prev_beg, (prev_end - prev_beg), prev_end, "speech"]
    smoothed_segments.append(smooth)
    return smoothed_segments


def smooth_segments(segments, gap):
    """Smoothing adjacent segments with same speaker label and occuring within a specified time."""
    # collect same speaker segments
    spkr_segments = {}
    for seg in segments:
        spkr = seg[3]
        if spkr not in spkr_segments:
            spkr_segments[spkr] = []
        spkr_segments[spkr].append(seg)
    # go through each group and merge where possible
    allspkrs_segments = []
    for spkr in spkr_segments:
        segments = spkr_segments[spkr]
        new_segments, prev = [], [-gap * 2, 0, -gap * 2, ""]
        prev_0 = prev
        for seg in segments:
            if seg[3] == prev[3]:    # if same speaker as previous segment
                segment_gap = seg[0] - prev[2]
                if 0 < segment_gap < gap:    # if segment_gap less than gap
                    dur = seg[2] - prev[0]
                    prev = [prev[0], dur] + seg[2:]
                else:    # if segment_gap larger than gap
                    if prev != prev_0:
                        new_segments.append(prev)
                    prev = seg
            else:    # if different speakers
                if prev != prev_0:
                    new_segments.append(prev)
                prev = seg
        if prev not in new_segments and prev != [-gap * 2, 0, -gap * 2, ""]:
            new_segments.append(prev)
        allspkrs_segments += new_segments
    # sort all by start time
    allspkrs_segments = sorted(allspkrs_segments, key=lambda seg: seg[0])
    return allspkrs_segments


def find_adjacent_segment_type(segments, time):
    """Find boundary type on left and right (NONSPEECH or SPEECH)"""
    # find previous segment type
    segments[0].append("NS")
    prev = segments[0]
    for i in range(1, len(segments)):
        if (segments[i][0] - time) > prev[2]:
            segments[i].append("NS")    # nonspeech
        else:
            segments[i].append("SC")    # speaker change
        prev = segments[i]
    # find following segment type
    for i in range(0, len(segments) - 1):
        if (segments[i][2] + time) < segments[i + 1][0]:
            segments[i].append("NS")    # nonspeech
        else:
            segments[i].append("SC")    # speaker change
    segments[len(segments) - 1].append("NS")
    return segments


########################################
# MATCH SEGMENTS
########################################


def match_segments(ref_segments, rSpkrs, hyp_segments, hSpkrs, args):
# matching hypothesis segments which fall within a reference segment boundaries -/+ collar
    """This function finds the matched segments

    Storing in a dictionary, each reference is checked to find a hypothesis segment whoses boundaries fall within start and end boundaries +/- collar of the reference. If boundary type is NONSPEECH, the collar can be scaled to give an asymetric distribution around the boundary. A hypothesis segment which has been matched to more than one reference is considered a duplicate and these are collected. Any unmatched hypothesis segments are found. If multiple hypothesis segments are selected for a single reference (unless there's a duplicated hypothesis segment), smooth those with same speaker label with specified time gap. For the 1-1 matches, carry out the match or no match decision based on the specified distribution (and threshold and padding around hypothesis boudnaries). Using these new 1-1 matches, carry out the speaker mapping. Using the new speaker map, duplicated hypothesis segments can be matched to a single reference given the speaker map, but if there is no speaker map for this segment label, choose by maximum time. Do a final smooth for these hypothesis segments and a final check on the boundaries. Collect the unmatched reference segments.
    """
    # dictionaries for reference labels, and for hypothesis to reference labels
    matched, matched_hyp = {}, {}
    # give segments a label and make put segments in dictionary
    ref_dict = make_dict(ref_segments)
    hyp_dict = make_dict(hyp_segments)
    # find matches for each reference segment
    for rlbl in ref_dict:
        [rbeg, rdur, rend, rspkr, rprevtype, rnexttype] = ref_dict[rlbl]
        # allowing extra wide boundary (unequal) if nonspeech
     #   extratimeprev, extratimenext = args.collar, args.collar
    #    if args.collar_scale != 1.0:
   #         if rprevtype == "NS":
  #              extratimeprev = args.collar * args.collar_scale
 #           if rnexttype == "NS":
#                extratimenext = args.collar * args.collar_scale
        for hlbl in hyp_dict:
            [hbeg, hdur, hend, hspkr, hprevtype, hnexttype] = hyp_dict[hlbl]
            if hlbl not in matched_hyp:
                matched_hyp[hlbl] = []
            if (rbeg - args.collar) <= hbeg and hend <= (rend + args.collar):
                matched_hyp[hlbl].append(rlbl)
    # find hypothesis segments matched to multiple reference segments
    hyp_multimatch = [hlbl for hlbl in matched_hyp if len(matched_hyp[hlbl]) > 1]
    # organise found matches
    for rlbl in ref_dict:
        matched[rlbl] = {}
        matched[rlbl]["REFSEG"] = ref_dict[rlbl]
        matched[rlbl]["LABELS"] = [hlbl for hlbl in matched_hyp if rlbl in matched_hyp[hlbl]]
        if matched[rlbl]["LABELS"] == []:
            matched[rlbl]["HYPSEG"] = []
        else:
            matched[rlbl]["HYPSEG"] = [[hseg for hseg in hyp_dict[hlbl]] for hlbl in matched[rlbl]["LABELS"]]
    # find unmatched hypothesis segments
    unmatched = {}
    unmatched["HYP"] = [[hseg for hseg in hyp_dict[hlbl]] for hlbl in hyp_dict if matched_hyp[hlbl] == []]
    # sorting out boundaries to find 1-1 matches
    for rlbl in matched:
        # multiple system segments found
        if len(matched[rlbl]["HYPSEG"]) > 1:
            types = []
            for hlbl in matched[rlbl]["LABELS"]:
                if hlbl in hyp_multimatch:
                    types.append("DUP")  # duplicate
                else:
                    types.append("HYP")
            # no duplicates found
            if "DUP" not in types:
                matched[rlbl]["HYPSEG"] = smooth_segments(matched[rlbl]["HYPSEG"], args.gap)
                matched[rlbl]["LABELS"] = ["notdup"] * len(matched[rlbl]["HYPSEG"])
                # remove if still >1 hyp segments after merging
                if len(matched[rlbl]["HYPSEG"]) > 1:
                    unmatched["HYP"] += matched[rlbl]["HYPSEG"]
                    matched[rlbl]["HYPSEG"] = []
                    matched[rlbl]["LABELS"] = []
        # only one system segment found
        if len(matched[rlbl]["HYPSEG"]) == 1:
            score = boundary_check(matched[rlbl]["REFSEG"], matched[rlbl]["HYPSEG"][0], args)
            if score < args.threshold:
                unmatched["HYP"] += matched[rlbl]["HYPSEG"]
                matched[rlbl]["HYPSEG"] = []
                matched[rlbl]["LABELS"] = []
    # find speaker pairs if not SAD
    if args.sad:
        spkr_pairs, SE, num_pairs, SSE, num_matched = {}, 0, 0, 0, 0
    else:
        [spkr_pairs, SE, num_pairs, SSE, num_matched] = map_spkrs(matched, rSpkrs, hSpkrs, hyp_multimatch, args)
    # fix multiples/duplicates
    if len(hyp_multimatch) > 0:
        matched = fix_duplicates(matched, hyp_multimatch, spkr_pairs)
    # final merge and check boundaries
    for rlbl in matched:
        matched[rlbl]["HYPSEG"] = smooth_segments(matched[rlbl]["HYPSEG"], args.gap)
        if len(matched[rlbl]["HYPSEG"]) > 1:
            unmatched["HYP"] += matched[rlbl]["HYPSEG"]
            matched[rlbl]["HYPSEG"] = []
            matched[rlbl]["LABELS"] = []
        if len(matched[rlbl]["HYPSEG"]) == 1:
            score = boundary_check(matched[rlbl]["REFSEG"], matched[rlbl]["HYPSEG"][0], args)
            if score < args.threshold:
                matched[rlbl]["HYPSEG"] = []
                matched[rlbl]["LABELS"] = []
    # find unmatched segments
    unmatched["REF"] = [[rseg for rseg in ref_dict[rlbl]] for rlbl in matched if matched[rlbl]["HYPSEG"] == []]
    return [matched, unmatched, spkr_pairs, SE, num_pairs, SSE, num_matched]


def make_dict(segments):
    """Create a dictionary giving an id number for each segment."""
    dict_segments = {}
    for seg in segments:
        dict_segments[segments.index(seg)] = seg
    return dict_segments


def boundary_check(refseg, hypseg, args):
    """Check if start and end lie within allowed reference boundary times."""
    rbeg, rend = refseg[0], refseg[1]
    hbeg, hend = hypseg[0], hypseg[1]
    score1 = match_start(rbeg, hbeg, args)
    if score1 != 0:
        score2 = match_end(rend, hend, args)
        return score1 * score2
    else:
        return 0


def match_start(ref, hyp, args):
    """Match start boundary given distribution."""
    score = 0.0
    collar = args.collar

    if args.distribution == "uniform":
        if (ref - collar < hyp and ref + collar > hyp):
            score = 1.0
    elif args.distribution == "triangular":
        a = hyp - ref
        b = 1. / (4 * collar * collar)

        if hyp + collar < ref - collar:  # A
            score = 0.0
        elif hyp + collar < ref:    # B
            score = 0.0
        elif hyp + collar < ref + collar:    # C
            score = (a + 2. * collar) * (a + 2. * collar) * b
        elif hyp == ref:    # D
            score = 1. - a * a * b
        elif hyp - collar < ref:    # E
            score = - (a - 4 * collar) * a * b
        elif hyp - collar < ref + collar:    # F
            score = 1. - (a - 4 * collar) * a * b
        else:  # G
            score = 0.0

    elif args.distribution == "Gaussian":
        from scipy.stats import norm
        score = norm.cdf(hyp + collar, ref, collar / 3) - \
                norm.cdf(hyp - collar, ref, collar / 3)
    return score


def match_end(ref, hyp, args):
    """Match end boundary given distribution."""
    score = 0.0
    collar = args.collar

    if args.distribution == "uniform":
        if (ref - collar < hyp and ref + collar > hyp):
            score = 1.0

    elif args.distribution == "triangular":
        a = hyp - ref
        b = 1. / (4 * collar * collar)

        if hyp + collar < ref - collar:  # A
            score = 0.0
        elif hyp + collar < ref:    # B
            score = 0.0
        elif hyp + collar < ref + collar:    # C
            score = (a + 2. * collar) * (a + 2. * collar) * b
        elif hyp == ref:    # D
            score = 1. - a * a * b
        elif hyp - collar < ref:    # E
            score = - (a - 4 * collar) * a * b
        elif hyp - collar < ref + collar:    # F
            score = 1. - (a - 4 * collar) * a * b
        else:  # G
            score = 0.0

    elif args.distribution == "Gaussian":
        from scipy.stats import norm
        score = norm.cdf(hyp + collar, ref, collar / 3) - \
                norm.cdf(hyp - collar, ref, collar / 3)
    return score


########################################
# SPEAKER MAPPING
########################################


def map_spkrs(matched, rSpkrs, hSpkrs, hyp_multimatch, args):
    """Map reference speakers to hypothesis clusters.

    This uses only the 1-1 reference to hypothesis matched segments. For each possible speaker-cluster pair, find their number of segments. If there are no matched segments, exit. Else, find the score for each pair using P(h=S1|r=SA,O)P(r=SA,O). For each hypothesis in the pairs found, choose the reference match which has the highest score. For these new hypothesis - reference pairs, find all possible combinations of the other hypothesis - reference pairs and sum the scores of the unselected pairs to use as an overall cost. Choose the combination with the lowest cost. Calculate whether each reference in the chosen combination is pure to the chosen hypothesis or impure (segments across other clusters). Calculate speaker error (sum of impure and unassigned reference speakers divided by all reference) and speaker segment error (sum of the segments from unchosen reference - hypothesis pairs divided by all 1-1 matched segments).
    """
    segs_info, total_segs, spkr_pairs, ref_spkrs, hyp_spkrs = {}, 0, {}, [], []
    # segment based speaker mapping
    for rlbl in matched:
        [rbeg, rdur, rend, rspkr, rprevtype, rnexttype] = matched[rlbl]["REFSEG"]
        if rspkr not in segs_info:
            segs_info[rspkr] = {}
            segs_info[rspkr]["TOTAL"] = 0
        # only consider 1-1 matches for speaker mapping
        if len(matched[rlbl]["LABELS"]) == 1:
            [hbeg, hdur, hend, hspkr, hprevtype, hnexttype] = matched[rlbl]["HYPSEG"][0]
            if hspkr not in segs_info[rspkr]:
                segs_info[rspkr][hspkr] = 1
            else:
                segs_info[rspkr][hspkr] += 1
            segs_info[rspkr]["TOTAL"] += 1
            total_segs += 1
    # if no segments have been matched
    num_matched = float(total_segs)
    if num_matched == 0:
        num_pairs, SE, SSE = 0, 100, 100
        print "REF SPEAKERS:\t\t\t%d" % (len(rSpkrs))
        print "HYP CLUSTERS:\t\t\t%d" % (len(hSpkrs))
        print "BOUNDARY MATCHED SEGMENTS:\t%d" % (num_matched)
        print "MAPPED SPEAKER-CLUSTER PAIRS:\t%d" % (num_pairs)
        print "UNASSIGNED SPEAKERS:\t\t%d" % (len(rSpkrs))
        print "UNASSIGNED CLUSTERS:\t\t%d" % (len(hSpkrs))
        print "PURE SPEAKER-CLUSTER PAIRS:\t%d" % (0)
        print "IMPURE SPEAKER-CLUSTER PAIRS:\t%d" % (0)
        print "SPEAKER ERROR (SE):\t\t%.2f %s" % (SE, "%")
        print "SPEAKER SEGMENT ERROR (SSE):\t%.2f %s ( %d / %d )" % (SSE, "%", 0, num_matched)
        return [{}, SE, num_pairs, SSE, num_matched]
    # find scores for each ref and hyp pair: P(h=S1|r=SA,O)P(r=SA,O)
    pairlist = []
    for rs in segs_info:
        for hs in segs_info[rs]:
            if hs != "TOTAL":
                num = segs_info[rs][hs]
                total = segs_info[rs]["TOTAL"]
                score = (float(num) / float(total)) * (float(total) / float(total_segs))
                pairlist.append([rs, hs, num, total, total, total_segs, score])
    if args.spkr_map:
        print "--------\nPAIR SCORES FOUND:"
        for p in pairlist:
            print p
        print "\nINITIAL HYP CLUSTER MATCHED TO REF SPEAKER:"
    pairlist = sorted(pairlist, key=lambda p: p[6], reverse=True)
    # from hypothesis point of view, can only have a single reference speaker match, take max score
    hyp_pairs = {}
    for hs in hSpkrs:
        hyp_pairs[hs] = ""
        max_score = 0
        for p in pairlist:
            if hs == p[1]:
                if max_score < p[6]:
                    hyp_pairs[hs] = p[0]
                    max_score = p[6]
    # check matches and select highest on ratio
    refs, pairs = [], []
    for hs in hyp_pairs:
        if args.spkr_map:
            print hs, hyp_pairs[hs]
        if hyp_pairs[hs] != "":
            refs.append(hyp_pairs[hs])
            pairs.append([hyp_pairs[hs], hs])
    # for every pair - check score (or cost) for choosing optimum combination of pairs
    errors = []
    for [rs1, hs1] in pairs:
        error = 0
        # with chosen pair, find possible other pairs
        sub_pairs = [[rs2, hs2] for [rs2, hs2] in pairs if [rs1, hs1] != [rs2, hs2] and rs1 != rs2 and hs1 != hs2] + [[rs1, hs1]]
        # check to see if these include the same references in some pairs, if not, carry on to error checking
        tmpref = [rs2 for [rs2, hs2] in sub_pairs]
        groups = []
        if len(tmpref) == len(set(tmpref)):
            groups.append(sub_pairs)
        else:
            # find non duplicates
            nodup = list(set([rs for rs in tmpref if tmpref.count(rs) == 1]))
            # find duplicates
            dup = list(set([rs for rs in tmpref if tmpref.count(rs) > 1]))
            # non duplicates
            group_base = [[rs1, hs1]]
            for rs in nodup:
                for [rs2, hs2] in sub_pairs:
                    if rs == rs2:
                        if [rs2, hs2] not in group_base:
                            group_base.append([rs2, hs2])
            # find pairs with duplicated references
            duppairs, lens = [], []
            for rs in dup:
                duppairs.append([p for p in sub_pairs if rs in p])
                lens.append(range(len([p for p in sub_pairs if rs in p])))
            # find combinations of duplicates
            combinations = []
            for l in lens:
                if combinations == []:
                    combinations = l
                else:
                    combinations = list(itertools.product(combinations, l))
            for i in combinations:
                comb = str(i).replace("(", " ").replace(")", " ").replace(",", " ").split()
                group_dup = []
                count = 0
                for i in comb:
                    group_dup.append(duppairs[count][int(i)])
                    count += 1
                groups.append(group_base + group_dup)
        # check all groups to find score/cost
        for group in groups:
            error = 0
            for p in pairlist:
                if [p[0], p[1]] not in group:
                    error += p[6]
        errors.append([group, error])
    # choose group with lowest score/cost
    if args.spkr_map:
        print "\nERROR WITH GROUP:"
        for [group, error] in sorted(errors, key=lambda e: e[1]):
            print error, "\t", group
    lowest_error = sorted(errors, key=lambda e: e[1])[0]
    for pair in lowest_error[0]:
        spkr_pairs[pair[0]] = pair[1]
    if args.spkr_map:
        print "\nFINAL SPEAKER-CLUSTER MAPPING:"
        for rs in spkr_pairs:
            print rs, "\t", spkr_pairs[rs]
    # speaker error and pure/impure speaker-cluster pairs
    pure, impure, spkr_seg_err = 0, 0, 0
    for rs in spkr_pairs:
        impurity = False
        hs = spkr_pairs[rs]
        #pure/impure speakers
        for p2 in pairlist:
            if rs == p2[0] and hs != p2[1]:
                impurity = True
                break
            if hs == p2[1] and rs != p2[0]:
                impurity = True
                break
        if impurity:
            impure += 1
        else:
            pure += 1
    # speaker segment error
    for p in pairlist:
        if p[0] not in spkr_pairs:
            spkr_seg_err += p[2]
        elif spkr_pairs[p[0]] != p[1]:
            spkr_seg_err += p[2]
    num_pairs = float(len(spkr_pairs))
    unassigned_spkrs = len(rSpkrs) - len(spkr_pairs)
    unassigned_clus = len(hSpkrs) - len(spkr_pairs)
    SE = (float(impure) + unassigned_spkrs) / float(len(rSpkrs)) * 100.0
    if total_segs != 0:
        SSE = float(spkr_seg_err) / float(total_segs) * 100.0
    else:
        SSE = 100
    print "--------"
    print "REF SPEAKERS:\t\t\t%d" % (len(rSpkrs))
    print "HYP CLUSTERS:\t\t\t%d" % (len(hSpkrs))
    print "BOUNDARY MATCHED SEGMENTS:\t%d" % (num_matched)
    print "MAPPED SPEAKER-CLUSTER PAIRS:\t%d" % (num_pairs)
    print "UNASSIGNED SPEAKERS:\t\t%d" % (unassigned_spkrs)
    print "UNASSIGNED CLUSTERS:\t\t%d" % (unassigned_clus)
    print "PURE SPEAKER-CLUSTER PAIRS:\t%d" % (pure)
    print "IMPURE SPEAKER-CLUSTER PAIRS:\t%d" % (impure)
    print "SPEAKER ERROR (SE):\t\t%.2f %s ( (%d + %d) / %d )" % (SE, "%", impure, unassigned_spkrs, len(rSpkrs))
    print "SPEAKER SEGMENT ERROR (SSE):\t%.2f %s ( %d / %d )" % (SSE, "%", spkr_seg_err, num_matched)
    return [spkr_pairs, SE, num_pairs, SSE, num_matched]


########################################
# FIX DUPLICATES BY SPKR LABELS OR TIME
########################################


def fix_duplicates(matched, hyp_multimatch, spkr_pairs):
    """For hypothesis segments matched to more than one reference segment, decide based on speaker mapping else the amount of maximum overlapping time."""
    for hlbl in hyp_multimatch:
        reflist = []
        for rlbl in matched:
            if hlbl in matched[rlbl]["LABELS"]:
                reflist.append([rlbl, matched[rlbl]["REFSEG"]])
                ind = matched[rlbl]["LABELS"].index(hlbl)
                hypseg = matched[rlbl]["HYPSEG"][ind]
        # decide which segment it belongs to
        remove = []
        for [lbl, seg] in reflist:
            if seg[3] in spkr_pairs:
                if spkr_pairs[seg[3]] != hypseg[3]:
                    remove.append(lbl)
            else:
                remove.append(lbl)
        # if no match by speaker label, choose by time overlap
        if len(remove) == len(reflist) and reflist != []:
            remove.pop(find_max_time_overlap(hypseg, reflist))
        # remove hyp segment from all unmatched reference segs
        for rlbl in remove:
            ind = matched[rlbl]["LABELS"].index(hlbl)
            matched[rlbl]["LABELS"].pop(ind)
            matched[rlbl]["HYPSEG"].pop(ind)
    return matched


def find_max_time_overlap(hypseg, reflist):
    """Find reference segment which encompasses the maximum time of the hypothesis segment."""
    hbeg, hend = hypseg[0], hypseg[2]
    times = []
    for [rlbl, rseg] in reflist:
        b = max(hbeg, rseg[0])
        e = min(hend, rseg[2])
        times.append(e - b)
    return times.index(max(times))


########################################
# EVALUATE MATCHED SEGMENTS USING FSEG
########################################


def evaluate_matched_segments(matched, unmatched, spkr_pairs, args):
    """Find segment matches, insertions and deletions."""
    NUMMAT, MAT, INS, DEL = 0, 0, len(unmatched["HYP"]), len(unmatched["REF"])
    for rlbl in matched:
        rspkr = matched[rlbl]["REFSEG"][3]
        # only one correct boundary match
        if len(matched[rlbl]["HYPSEG"]) == 1:
            hspkr = matched[rlbl]["HYPSEG"][0][3]
            # speech only so ignore speaker labels
            if args.sad:
                MAT += 1
            else:
                # evaluate speaker labels
                if rspkr in spkr_pairs:
                    # segment has correct speaker
                    if hspkr == spkr_pairs[rspkr]:
                        MAT += 1
                    # segment has incorrect speaker
                    else:
                        DEL += 1
                        INS += 1
                # segment has incorrect speaker
                else:
                    DEL += 1
                    INS += 1
    return [MAT, INS, DEL]


########################################
# SCORING
########################################


def score_single(MAT, INS, DEL, SE, num_pairs, SSE, num_matched):
    """Score a single file.

    This finds the number of reference segments, the percentages of the matches, insertions and deletions, calculates the segment F-measure and prints the scores.
    """
    numref = MAT + DEL
    [MAT, INS, DEL] = get_percentages(MAT, INS, DEL, numref)
    [PRC, RCL, F] = get_fmeasure(MAT, INS, DEL)
    print_scores(SE, SSE, MAT, INS, DEL, PRC, RCL, F)
    return [SE, SSE, MAT, INS, DEL, PRC, RCL, F, num_pairs, num_matched, numref]


def get_percentages(MAT, INS, DEL, numref):
    """Calculates percentages given number of reference segments."""
    MAT = float(MAT) / float(numref) * 100.0
    INS = float(INS) / float(numref) * 100.0
    DEL = float(DEL) / float(numref) * 100.0
    return [MAT, INS, DEL]


def get_fmeasure(MAT, INS, DEL):
    """Calculates the segment-based F-measure."""
    PRC = float(MAT) / float(MAT + INS)
    RCL = float(MAT) / float(MAT + DEL)
    if (PRC + RCL) != 0:
        F = 2 * ((PRC * RCL) / (PRC + RCL))
    else:
        F = 0
    return [PRC * 100.0, RCL * 100.0, F * 100.0]


def print_scores(SE, SSE, MAT, INS, DEL, PRC, RCL, F):
    """Prints the scores."""
    print "--------"
    print "SE:\t%.1f %s" % (SE, "%")
    print "SSE:\t%.1f %s" % (SSE, "%")
    print "MAT:\t%.1f %s" % (MAT, "%")
    print "INS:\t%.1f %s" % (INS, "%")
    print "DEL:\t%.1f %s" % (DEL, "%")
    print "PRC:\t%.1f %s" % (PRC, "%")
    print "RCL:\t%.1f %s" % (RCL, "%")
    print "F:\t%.1f %s" % (F, "%")


def find_total(scores, args):
    """Finds overall scores weighted by number of reference speakers for the SE, number of matched segments for the SSE and number of reference segments for the rest."""
    sums = [0] * len(scores[0])
    for [SE, SSE, MAT, INS, DEL, PRC, RCL, F, num_pairs, num_matched, numref] in scores:
        sums[0] += SE * num_pairs
        sums[1] += SSE * num_matched
        sums[2] += MAT * numref
        sums[3] += INS * numref
        sums[4] += DEL * numref
        sums[5] += PRC * numref
        sums[6] += RCL * numref
        sums[7] += F * numref
        sums[8] += num_pairs
        sums[9] += num_matched
        sums[10] += numref
    total_segs = sums[10]
    # sad only
    if args.sad:
        print_scores(0, 0, sums[2] / total_segs, sums[3] / total_segs, sums[4] / total_segs, sums[5] / total_segs, sums[6] / total_segs, sums[7] / total_segs)
    # no segments matched across files
    elif sums[0] == 0:
        print_scores(100, 100, sums[2] / total_segs, sums[3] / total_segs, sums[4] / total_segs, sums[5] / total_segs, sums[6] / total_segs, sums[7] / total_segs)
    else:
        print_scores(sums[0] / sums[8], sums[1] / sums[9], sums[2] / total_segs, sums[3] / total_segs, sums[4] / total_segs, sums[5] / total_segs, sums[6] / total_segs, sums[7] / total_segs)


########################################
# SAVING - CREATE RTTM FOR COMPARISON WITH OTHER METRICS
########################################


def saving_rttm(fname, matched, unmatched, args):
    """Creates rttms for use with other scoring metrics and saves to specified folder."""
    # write new smoothed segments to rttm files
    rlines, hlines = [], []
    hypsegs = unmatched["HYP"]
    for rlbl in matched:
        seg = matched[rlbl]["REFSEG"]
        rlines.append("SPEAKER\t%s\t1\t%.2f\t%.2f\t<NA>\t<NA>\t%s\t<NA>" % (fname, seg[0], seg[1], seg[3]))
        if matched[rlbl]["HYPSEG"] != []:
            hypsegs.append(matched[rlbl]["HYPSEG"][0])
    sorted_hypsegs = sorted(hypsegs, key=lambda seg: seg[0])
    for seg in sorted_hypsegs:
        hlines.append("SPEAKER\t%s\t1\t%.2f\t%.2f\t<NA>\t<NA>\t%s\t<NA>" % (fname, seg[0], seg[1], seg[3]))
    # print reference to file
    refname = args.folder + fname + ".ref.rttm"
    commands.getoutput("rm %s" % refname)
    with open(refname, 'w') as f:
        for l in rlines:
            print >> f, l
    # print hypothesis to file
    hypname = args.folder + fname + ".sys.rttm"
    commands.getoutput("rm %s" % hypname)
    with open(hypname, 'w') as f:
        for l in hlines:
            print >> f, l
    print "--------"
    print "FINAL# REF SEGS:\t", len(rlines)
    print "FINAL# SYS SEGS:\t", len(hlines)
    return [refname, hypname]


########################################
# ARGPARSE
########################################


def parser_setup():
    """Creates the parser for reading from the command line and creating a help section."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # compulsory files
    parser.add_argument("ref", help="Reference RTTM file (or list with flag --list)")
    parser.add_argument("hyp", help="Hypothesis RTTM file (or list with flag --list)")
    # optional files
    parser.add_argument("-u", "--uem", help="UEM file (can be single file or list)")
    # optional arguments
    parser.add_argument("-g", "--gap", help="Smoothing time gap between same speaker segments (seconds)", type=float, default=0.25)
    parser.add_argument("-c", "--collar", help="Collar around reference boundaries, +/- (seconds)", type=float, default=0.0)
    parser.add_argument("-d", "--distribution", help="Distribution around reference boundaries", type=str, choices=["uniform", "triangular", "Gaussian"], default="uniform")
    parser.add_argument("-t", "--threshold", help="Threshold for segment match/no match decision (not uniform distribution)", type=float, default=0.00005)
    parser.add_argument("-p", "--padding", help="Padding around hypothesis boundary (not uniform distribution)", type=float, default=0.005)
    parser.add_argument("-cs", "--collar-scale", help="Scale to multiply collar if boundary type NONSPEECH, for asymetric boundary distributions", type=float, default=1.0)
    parser.add_argument("-f", "--folder", help="Folder in which to save smoothed RTTM files", default="./")
    # optional flags
    parser.add_argument("--sad", help="Score Speech Activity Detection only, ignore speaker labels", action="store_true")
    parser.add_argument("-m", "--spkr-map", help="Display speaker mapping information", action="store_true")
    parser.add_argument("--list", help="REF and HYP are lists of RTTMs", action="store_true")
    parser.add_argument("--save", help="Save smoothed RTTM files", action="store_true")
    return parser.parse_args()


########################################
# MAIN
########################################


def main():
    # read from command line
    args = parser_setup()

    # setup for file lists, reading rttms, reading uem and creating output folder
    [reflines, hyplines, uem_times, args] = initial_setup(args)

    # for each pair of files
    scores = []
    for name in reflines:
        if name in hyplines:
            print "--------------------------------\nFILE:\t%s" % name

            # reading segments, applying uem times, smoothing segments, merging if SAD, boundary info
            [refsegments, refspkrs, hypsegments, hypspkrs, args] = organising_segments(name, reflines[name], hyplines[name], uem_times, args)

            # match segments, speaker pairs, sort out hypothesis segments matched to > 1 reference segment
            print "COLLAR:\t %.2f" % args.collar
            [matched, unmatched, spkr_pairs, SE, num_pairs, SSE, num_matched] = match_segments(refsegments, refspkrs, hypsegments, hypspkrs, args)

            # evaluate final segment matches
            [MAT, INS, DEL] = evaluate_matched_segments(matched, unmatched, spkr_pairs, args)

            # scoring
            scores.append(score_single(MAT, INS, DEL, SE, num_pairs, SSE, num_matched))

            # save files
            if args.save:
                saving_rttm(name, matched, unmatched, args)

    # overall scores
    if len(scores) > 1:
        print "--------------------------------\nOVERALL"
        find_total(scores, args)

if __name__ == "__main__":
    main()
