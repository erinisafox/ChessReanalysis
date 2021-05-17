from models import *
import chess, chess.pgn
import json
from collections import defaultdict
from datetime import datetime
from operator import eq, lt
from functools import partial
import math
from random import *

# TODO: Make this configurable via the config file.
_cp_loss_intervals = [0, 10, 25, 50, 100, 200, 500]
_cp_loss_names = ["=0"] + [f">{cp_incr}" for cp_incr in _cp_loss_intervals]
_cp_loss_ops = [partial(eq,0)] + [partial(lt, cp_incr) for cp_incr in _cp_loss_intervals]

def generate_stats_string(sample, total):
    percentage = sample / total
    stderr = std_error(percentage, total)
    ci = wilson_interval(sample, total)
    return f'{sample}/{total}; {percentage:.01%} (CI: {ci[0]*100:.01f} - {ci[1]*100:.01f})'

def generate_stats_string_csv(sample, total):
    percentage = sample / total
    stderr = std_error(percentage, total)
    ci = confidence_interval(percentage, stderr)
    return f'{sample}/{total},{percentage:.01%},{ci[0]*100:.01f},{ci[1]*100:.01f}'

def std_error(p, n):
    return math.sqrt(p*(1-p)/n)

# based on normal distribution, better to use wilson_interval defined below.
def confidence_interval(p, e):
    return [
        p - (2.0*e),
        p + (2.0*e)
    ]

#Empirical bootstrap interval - takes sample array and returns 95% CI
#BCa is a work in progress
def bootstrap_interval(array): #assumes 95% intervals as a convention
    mn = (1.0*sum(array)) / len(array)
    deltas = []
    for X in range(0,400): #400 bootstrap samples for convenience
        #Calculating the average while the sampling occurs.
        #The sample isn't ever explicitly generated.
        bootsum = 0.0
        for Y in range(0,len(array)):
            bootsum += array[randrange(len(array))]
        deltas.append((bootsum / len(array)) - mn)
    deltas = sorted(deltas)
    lowerval = deltas[9] #should correspond to the 2.5th percentile
    upperval = deltas[389] #should correspond to the 97.5th percentile
    
    return [round(mn - upperval,1), round(mn - lowerval,1)]

class PgnSpyResult():

    def __init__(self):
        self.sample_size = 0
        self.sample_total_cpl = 0
        self.gt0 = 0
        self.gt10 = 0
        self.t1_total = 0
        self.t1_count = 0
        self.t2_total = 0
        self.t2_count = 0
        self.t3_total = 0
        self.t3_count = 0

        self.wt1_total = 0
        self.wt1_count = 0
        self.wt2_total = 0
        self.wt2_count = 0
        self.wt3_total = 0
        self.wt3_count = 0

        self.bt1_total = 0
        self.bt1_count = 0
        self.bt2_total = 0
        self.bt2_count = 0
        self.bt3_total = 0
        self.bt3_count = 0
		
        self.min_rating = None
        self.max_rating = None
        self.game_list = []
        self.cp_loss_count = defaultdict(int)
        self.cp_loss_total = 0
        self.cp_loss_list_by_move = []
        self.cp_loss_list_by_game = []

    def add(self, other):
        self.sample_size += other.sample_size
        self.sample_total_cpl += other.sample_total_cpl
        self.gt0 += other.gt0
        self.gt10 += other.gt10
        
        self.t1_total += other.t1_total
        self.t1_count += other.t1_count
        self.t2_total += other.t2_total
        self.t2_count += other.t2_count
        self.t3_total += other.t3_total
        self.t3_count += other.t3_count
        
        self.wt1_total += other.wt1_total
        self.wt1_count += other.wt1_count
        self.wt2_total += other.wt2_total
        self.wt2_count += other.wt2_count
        self.wt3_total += other.wt3_total
        self.wt3_count += other.wt3_count

        self.bt1_total += other.bt1_total
        self.bt1_count += other.bt1_count
        self.bt2_total += other.bt2_total
        self.bt2_count += other.bt2_count
        self.bt3_total += other.bt3_total
        self.bt3_count += other.bt3_count
        ####
        self.with_rating(other.min_rating)
        self.with_rating(other.max_rating)
        self.game_list += other.game_list
        for k in _cp_loss_names:
            self.cp_loss_count[k] += other.cp_loss_count[k]
        self.cp_loss_total += other.cp_loss_total
        self.cp_loss_list_by_move += other.cp_loss_list_by_move #gives a 1d array with CPLs for each move
        self.cp_loss_list_by_game.append(other.cp_loss_list_by_game) #gives 2d array with CPLs in individual games

    def with_rating(self, rating):
        if rating is None:
            return
        self.min_rating = min(self.min_rating, rating) if self.min_rating else rating
        self.max_rating = max(self.max_rating, rating) if self.max_rating else rating

    @property
    def acpl(self):
        return self.sample_total_cpl / float(self.sample_size) if self.sample_size else None

    @property
    def t3_sort(self):
        if self.t3_total == 0:
            return 0
        return -wilson_interval(self.t3_count, self.t3_total)[0]

def t_output(fout, result):
    #Commented some of the less important output to make it easier on the eyes
    if result.t1_total:
        fout.write('T1: {}\n'.format(generate_stats_string(result.t1_count, result.t1_total)))
    if result.t2_total:
        fout.write('T2: {}\n'.format(generate_stats_string(result.t2_count, result.t2_total)))
    if result.t3_total:
        fout.write('T3: {}\n'.format(generate_stats_string(result.t3_count, result.t3_total)))
    if result.wt1_total:
        fout.write('WT1: {}\n'.format(generate_stats_string(result.wt1_count, result.wt1_total)))
    #if result.wt2_total:
    #    fout.write('WT2: {}\n'.format(generate_stats_string(result.wt2_count, result.wt2_total)))
    #if result.wt3_total:
    #    fout.write('WT3: {}\n'.format(generate_stats_string(result.wt3_count, result.wt3_total)))
    if result.bt1_total:
        fout.write('BT1: {}\n'.format(generate_stats_string(result.bt1_count, result.bt1_total)))
    #if result.bt2_total:
    #    fout.write('BT2: {}\n'.format(generate_stats_string(result.bt2_count, result.bt2_total)))
    #if result.bt3_total:
    #    fout.write('BT3: {}\n'.format(generate_stats_string(result.bt3_count, result.bt3_total)))
    
    if result.acpl:
        game_acpl = [] #temp variable to help with averaging ACPLs of games
        for X in range(0,len(result.cp_loss_list_by_game)):
            if len(result.cp_loss_list_by_game[X]):
                game_acpl.append(sum(result.cp_loss_list_by_game[X])/len(result.cp_loss_list_by_game[X]))
        #Print ACPL calculated from average CPLs of moves
        fout.write(f'ACPL M: {result.acpl:.1f} ({result.sample_size}) {str(bootstrap_interval(result.cp_loss_list_by_move))}\n')
        #Print ACPL calculated from average ACPLs of games, if possible
        if len(game_acpl):
            fout.write(f'ACPL G: {sum(game_acpl)/len(game_acpl):.1f} ({len(game_acpl)}) {str(bootstrap_interval(game_acpl))}\n')
    total = result.cp_loss_total
    if total > 0:
        for cp_loss_name in _cp_loss_names:
            loss_count = result.cp_loss_count[cp_loss_name]
            stats_str = generate_stats_string(loss_count, total)
            fout.write(f'  {cp_loss_name} CP loss: {stats_str}\n')

#Character spam since I don't know any better ways to do it
#I never use CSV exports, so this isn't priority for me to tidy up
def t_output_csv(fout, result):
    #Commenting out lesser used stats to make it easier on the eyes
    if result.t1_total:
        fout.write(f'{result.t1_count}/{result.t1_total},{result.t1_count / result.t1_total:.1%},')
    else:
        fout.write('x,x,')
    if result.t2_total:
        fout.write(f'{result.t2_count}/{result.t2_total},{result.t2_count / result.t2_total:.1%},')
    else:
        fout.write('x,x,')
    if result.t3_total:
        fout.write(f'{result.t3_count}/{result.t3_total},{result.t3_count / result.t3_total:.1%},')
    else:
        fout.write('x,x,')
    if result.wt1_total:
        fout.write(f'{result.wt1_count}/{result.wt1_total},{result.wt1_count / result.wt1_total:.1%},')
    else:
        fout.write('x,x,')
    #if result.wt2_total:
    #    fout.write(f'{result.wt2_count}/{result.wt2_total},{result.wt2_count / result.wt2_total:.1%},')
    #else:
    #    fout.write('x,x,')
    #if result.wt3_total:
    #    fout.write(f'{result.wt3_count}/{result.wt3_total},{result.wt3_count / result.wt3_total:.1%},')
    #else:
    #    fout.write('x,x,')
    if result.bt1_total:
        fout.write(f'{result.bt1_count}/{result.bt1_total},{result.bt1_count / result.bt1_total:.1%},')
    else:
        fout.write('x,x,')
    #if result.bt2_total:
    #    fout.write(f'{result.bt2_count}/{result.bt2_total},{result.bt2_count / result.bt2_total:.1%},')
    #else:
    #    fout.write('x,x,')
    #if result.bt3_total:
    #    fout.write(f'{result.bt3_count}/{result.bt3_total},{result.bt3_count / result.bt3_total:.1%},')
    #else:
    #    fout.write('x,x,')
    if result.acpl:
        #Print ACPL calculated from average CPLs of moves
        fout.write(f'{result.acpl:.1f},{result.sample_size},{str(bootstrap_interval(result.cp_loss_list_by_move))},')
    else:
        fout.write('x,x,x,x,')
    #Print ACPL calculated from average ACPLs of games, if possible
    game_acpl = [] #temp variable to help with averaging ACPLs of games
    for X in range(0,len(result.cp_loss_list_by_game)):
        if len(result.cp_loss_list_by_game[X]):
            game_acpl.append(sum(result.cp_loss_list_by_game[X])/len(result.cp_loss_list_by_game[X]))
    if len(game_acpl):
        fout.write(f'{sum(game_acpl)/len(game_acpl):.1f},{len(game_acpl)},{str(bootstrap_interval(game_acpl))},')
    else:
        fout.write('x,x,x,x,')

    total = result.cp_loss_total
    if total > 0:
        for cp_loss_name in _cp_loss_names:
            loss_count = result.cp_loss_count[cp_loss_name]
            stats_str = generate_stats_string_csv(loss_count, total)
            fout.write(f'{stats_str},')
    else:
        for cp_loss_name in _cp_loss_names:
            fout.write(f',,,,')

def a1(working_set, report_name):
    p = load_a1_params()
    by_player = defaultdict(PgnSpyResult)
    by_game = defaultdict(PgnSpyResult)
    excluded = included = 0
    for gid, pgn in working_set.items():
        game_obj, _ = Game.get_or_create(id=gid)
        if not game_obj.is_analyzed:
            excluded += 1
            continue

        a1_game(p, by_player, by_game, game_obj, pgn, 'w', GamePlayer.get(game=game_obj, color='w').player)
        a1_game(p, by_player, by_game, game_obj, pgn, 'b', GamePlayer.get(game=game_obj, color='b').player)
        included += 1
    if excluded:
        print(f'Skipping {excluded} games that haven\'t been pre-processed')

    out_path = f'reports/report-a1--{datetime.now():%Y-%m-%d--%H-%M-%S}--{report_name}.txt'
    with open(out_path, 'w') as fout:
        fout.write('------ BY PLAYER ------\n\n')
        for player, result in sorted(by_player.items(), key=lambda i: i[1].t3_sort):
            fout.write(f'{player.username} ({result.min_rating} - {result.max_rating})\n')
            t_output(fout, result)
            fout.write(' '.join(result.game_list) + '\n')
            fout.write('\n')

        fout.write('\n------ BY GAME ------\n\n')
        for (player, gameid), result in sorted(by_game.items(), key=lambda i: i[1].t3_sort):
            fout.write(f'{player.username} ({result.min_rating})\n')
            t_output(fout, result)
            fout.write(' '.join(result.game_list) + '\n')
            fout.write('\n')
    print(f'Wrote report on {included} games to "{out_path}"')

def a1csv(working_set, report_name):
    p = load_a1_params()
    by_player = defaultdict(PgnSpyResult)
    by_game = defaultdict(PgnSpyResult)
    excluded = included = 0
    for gid, pgn in working_set.items():
        game_obj, _ = Game.get_or_create(id=gid)
        if not game_obj.is_analyzed:
            excluded += 1
            continue

        a1_game(p, by_player, by_game, game_obj, pgn, 'w', GamePlayer.get(game=game_obj, color='w').player)
        a1_game(p, by_player, by_game, game_obj, pgn, 'b', GamePlayer.get(game=game_obj, color='b').player)
        included += 1
    if excluded:
        print(f'Skipping {excluded} games that haven\'t been pre-processed')

    out_path = f'reports/report-a1--{datetime.now():%Y-%m-%d--%H-%M-%S}--{report_name}.csv'
    with open(out_path, 'w') as fout:
        cp_loss_name_string = ''
        for cp_loss_name in _cp_loss_names:
            cp_loss_name_string += f'CPL{cp_loss_name},CPL{cp_loss_name}%,CPL{cp_loss_name} CI lower,CPL{cp_loss_name} CI upper,'
        fout.write(f'Name,Rating range,T1:,T1%:,T2:,T2%:,T3:,T3%:,WT1:,WT1%,BT1:,BT1%:,ACPL M:,Positions,LCI,UCI,ACPL G,Positions,LCI,UCI,{cp_loss_name_string}Games\n')
        for player, result in sorted(by_player.items(), key=lambda i: i[1].t3_sort):
            fout.write(f'{player.username},{result.min_rating} - {result.max_rating},')
            t_output_csv(fout, result)
            fout.write(' '.join(result.game_list) + '\n')

    print(f'Wrote report on {included} games to "{out_path}"')

def a1_game(p, by_player, by_game, game_obj, pgn, color, player):
    moves = list(Move.select().where(Move.game == game_obj).order_by(Move.number, -Move.color))

    r = PgnSpyResult()
    r.game_list.append(game_obj.id)
    try:
        r.with_rating(int(pgn.headers['WhiteElo' if color == 'w' else 'BlackElo']))
    except ValueError:
        pass
    evals = []
    for m in moves:
        
        if m.color != color:
            evals.append(-m.pv1_eval)
            continue
        evals.append(m.pv1_eval)

        if m.number <= p['book_depth']:
            continue
		
	#Start of major edits. Trying to make my dreams come true
        #Tracks negative eval positions.
        if m.pv1_eval > p['undecided_pos_thresh'] and m.pv1_eval <= 99999:
            if m.pv2_eval is not None and m.pv1_eval <= m.pv2_eval + p['forced_move_thresh'] and m.pv1_eval <= m.pv2_eval + p['unclear_pos_thresh']:
                if m.pv2_eval < m.pv1_eval:
                    r.bt1_total += 1
                    if m.played_rank and m.played_rank <= 1:
                        r.bt1_count += 1
                if m.pv3_eval is not None and m.pv2_eval <= m.pv3_eval + p['forced_move_thresh'] and m.pv1_eval <= m.pv3_eval + p['unclear_pos_thresh']:
                    if m.pv3_eval < m.pv2_eval:
                        r.bt2_total += 1
                        if m.played_rank and m.played_rank <= 2:
                            r.bt2_count += 1
                    if m.pv4_eval is not None and m.pv3_eval <= m.pv4_eval + p['forced_move_thresh'] and m.pv1_eval <= m.pv4_eval + p['unclear_pos_thresh']:
                        if m.pv4_eval < m.pv3_eval:
                            r.bt3_total += 1
                            if m.played_rank and m.played_rank <= 3:
                                r.bt3_count += 1
            continue
        
	#Second stage of major edits
        #Tracks positive eval positions (not as useful, probably)
        if m.pv1_eval < -p['undecided_pos_thresh'] and m.pv1_eval >= -99999:
            if m.pv2_eval is not None and m.pv1_eval <= m.pv2_eval + p['forced_move_thresh'] and m.pv1_eval <= m.pv2_eval + p['unclear_pos_thresh']:
                if m.pv2_eval < m.pv1_eval:
                    r.wt1_total += 1
                    if m.played_rank and m.played_rank <= 1:
                        r.wt1_count += 1
                if m.pv3_eval is not None and m.pv2_eval <= m.pv3_eval + p['forced_move_thresh'] and m.pv1_eval <= m.pv3_eval + p['unclear_pos_thresh']:
                    if m.pv3_eval < m.pv2_eval:
                        r.wt2_total += 1
                        if m.played_rank and m.played_rank <= 2:
                            r.wt2_count += 1
                    if m.pv4_eval is not None and m.pv3_eval <= m.pv4_eval + p['forced_move_thresh'] and m.pv1_eval <= m.pv4_eval + p['unclear_pos_thresh']:
                        if m.pv4_eval < m.pv3_eval:
                            r.wt3_total += 1
                            if m.played_rank and m.played_rank <= 3:
                                r.wt3_count += 1
            continue

	#Basically what it was originally
        #Tracks undecided positions
        if abs(m.pv1_eval) <= p['undecided_pos_thresh']:
            if m.pv2_eval is not None and m.pv1_eval <= m.pv2_eval + p['forced_move_thresh'] and m.pv1_eval <= m.pv2_eval + p['unclear_pos_thresh']:
                if m.pv2_eval < m.pv1_eval:
                    r.t1_total += 1
                    if m.played_rank and m.played_rank <= 1:
                        r.t1_count += 1
                if m.pv3_eval is not None and m.pv2_eval <= m.pv3_eval + p['forced_move_thresh'] and m.pv1_eval <= m.pv3_eval + p['unclear_pos_thresh']:
                    if m.pv3_eval < m.pv2_eval:
                        r.t2_total += 1
                        if m.played_rank and m.played_rank <= 2:
                            r.t2_count += 1
                    if m.pv4_eval is not None and m.pv3_eval <= m.pv4_eval + p['forced_move_thresh'] and m.pv1_eval <= m.pv4_eval + p['unclear_pos_thresh']:
                        if m.pv4_eval < m.pv3_eval:
                            r.t3_total += 1
                            if m.played_rank and m.played_rank <= 3:
                                r.t3_count += 1
        #Should be foolproof to do it like this?
        #Only go on to calculate ACPLs if undecided position
        #So that old results still have the same ACPL calculations
        else:
            continue

        initial_cpl = max(m.pv1_eval - m.played_eval, 0)
        r.cp_loss_total += 1
        for cp_name, cp_op in zip(_cp_loss_names, _cp_loss_ops):
            if cp_op(initial_cpl):
                r.cp_loss_count[cp_name] += 1

        cpl = min(max(m.pv1_eval - m.played_eval, 0), p['max_cpl'])
        if p['exclude_flat'] and cpl == 0 and evals[-3:] == [m.pv1_eval] * 3:
            # Exclude flat evals from CPL, e.g. dead drawn endings
            continue

        r.sample_size += 1
        r.sample_total_cpl += cpl
        r.cp_loss_list_by_move.append(cpl)
        r.cp_loss_list_by_game.append(cpl)
        #Have so many questions about this part
        if cpl > 0:
            r.gt0 += 1
        if cpl > 10:
            r.gt10 += 1

    by_player[player].add(r)
    by_game[(player, game_obj.id)].add(r)

def load_a1_params():
    with open('./config/params_for_a1.json') as config_f:
        return json.load(config_f)

def wilson_interval(ns, n):
    z = 1.95996 # 0.95 confidence
    a = 1 / (n + z**2)
    b = ns + z**2 / 2
    c = z * (ns * (n - ns) / n + z**2 / 4)**(1/2)
    return (a * (b - c), a * (b + c))
