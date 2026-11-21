// Push-optimal Sokoban solver with per-room decomposition.
// State space is pushes (not steps); walking is reconstructed by BFS and is free to the score.
// Rooms are independent because a box pushed into the hallway can never return to its room.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const CAP = 400000;               // deterministic expansion cap per room
const DIRS: [number, number, string][] = [
  [0, -1, 'u'], [0, 1, 'd'], [-1, 0, 'l'], [1, 0, 'r'],
];

interface Level {
  w: number; h: number;
  wall: boolean[];
  goals: Set<number>;
  boxes: number[];       // all boxes, sorted
  player: number;
}

function parse(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const w = Math.max(...rows.map((r) => r.length));
  const h = rows.length;
  const wall = new Array<boolean>(w * h).fill(true);
  const goals = new Set<number>();
  const boxes: number[] = [];
  let player = 0;
  for (let y = 0; y < h; y++) {
    const row = rows[y]!;
    for (let x = 0; x < w; x++) {
      const ch = row[x] ?? '#';
      const i = y * w + x;
      wall[i] = ch === '#';
      if ('.+*'.includes(ch)) goals.add(i);
      if ('$*'.includes(ch)) boxes.push(i);
      if ('@+'.includes(ch)) player = i;
    }
  }
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

// Connected component labels over non-wall cells, restricted to a band of rows.
function comps(lv: Level, r0: number, r1: number): Int32Array {
  const { w, h, wall } = lv;
  const label = new Int32Array(w * h).fill(-1);
  let n = 0;
  const inBand = (i: number): boolean => {
    const r = Math.floor(i / w);
    return r >= r0 && r < r1 && !wall[i]!;
  };
  for (let i = 0; i < w * h; i++) {
    if (!inBand(i) || label[i] !== -1) continue;
    const q = [i]; label[i] = n;
    for (let hq = 0; hq < q.length; hq++) {
      const c = q[hq]!;
      for (const [dx, dy] of DIRS) {
        const nx = (c % w) + dx, ny = Math.floor(c / w) + dy;
        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
        const j = ny * w + nx;
        if (inBand(j) && label[j] === -1) { label[j] = n; q.push(j); }
      }
    }
    n++;
  }
  return label;
}

interface Group { idx: number[]; goals: number[]; }

function groupsOf(lv: Level): Group[] {
  const hr = Math.floor(lv.player / lv.w);
  const up = comps(lv, 0, hr);
  const down = comps(lv, hr + 1, lv.h);
  const bandLabel = (i: number): string => {
    const r = Math.floor(i / lv.w);
    if (r < hr) return 'U' + up[i];
    if (r > hr) return 'D' + down[i];
    return 'H' + i;
  };
  const map = new Map<string, { idx: number[]; goals: number[] }>();
  lv.boxes.forEach((b, k) => {
    const g1 = bandLabel(b);
    let g = map.get(g1);
    if (!g) { g = { idx: [], goals: [] }; map.set(g1, g); }
    g.idx.push(k);
  });
  for (const gl of lv.goals) {
    const g1 = bandLabel(gl);
    let g = map.get(g1);
    if (!g) { g = { idx: [], goals: [] }; map.set(g1, g); }
    g.goals.push(gl);
  }
  return [...map.values()].filter((g) => g.idx.length > 0);
}

// BFS reachability of the player over cells that are neither wall nor occupied by a box.
function reach(lv: Level, boxes: Set<number>, from: number): { seen: Set<number>; parent: Map<number, number> } {
  const seen = new Set<number>([from]);
  const parent = new Map<number, number>();
  const q = [from];
  const { w, h, wall } = lv;
  for (let hq = 0; hq < q.length; hq++) {
    const c = q[hq]!;
    for (const [dx, dy] of DIRS) {
      const nx = (c % w) + dx, ny = Math.floor(c / w) + dy;
      if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
      const j = ny * w + nx;
      if (wall[j] || boxes.has(j) || seen.has(j)) continue;
      seen.add(j); parent.set(j, c); q.push(j);
    }
  }
  return { seen, parent };
}

function walkPath(from: number, to: number, parent: Map<number, number>, w: number): string {
  let cur = to; let path = '';
  while (cur !== from) {
    const p = parent.get(cur)!;
    const dx = (cur % w) - (p % w), dy = Math.floor(cur / w) - Math.floor(p / w);
    const d = DIRS.find(([a, b]) => a === dx && b === dy)!;
    path = d[2] + path; cur = p;
  }
  return path;
}

function manhattan(a: number, b: number, w: number): number {
  return Math.abs((a % w) - (b % w)) + Math.abs(Math.floor(a / w) - Math.floor(b / w));
}

// Minimum-cost assignment of boxes to goals (small groups; brute force permutations).
function heuristic(bs: number[], goals: number[], w: number): number {
  if (goals.length === 0) return 0;
  const n = bs.length;
  // Brute-force assignment is exact for the generator's rooms of 3-4 boxes; fall back to a
  // cheaper admissible bound if a group is unexpectedly large.
  if (n > 6 || goals.length > 6) {
    let s = 0; for (const b of bs) s += Math.min(...goals.map((g) => manhattan(b, g, w)));
    return s;
  }
  const cost: number[][] = bs.map((b) => goals.map((g) => manhattan(b, g, w)));
  let best = Infinity;
  const perm = (idx: number, used: boolean[], acc: number): void => {
    if (acc >= best) return;
    if (idx === n) { if (acc < best) best = acc; return; }
    for (let j = 0; j < goals.length; j++) {
      if (used[j]) continue;
      used[j] = true; perm(idx + 1, used, acc + cost[idx]![j]!); used[j] = false;
    }
  };
  perm(0, new Array(goals.length).fill(false), 0);
  return best === Infinity ? 0 : best;
}

function cornerDead(b: number, lv: Level, goals: Set<number>): boolean {
  if (goals.has(b)) return false;
  const { w, h, wall } = lv;
  const x = b % w, y = Math.floor(b / w);
  const up = y === 0 || wall[b - w]!;
  const down = y === h - 1 || wall[b + w]!;
  const left = x === 0 || wall[b - 1]!;
  const right = x === w - 1 || wall[b + 1]!;
  return (up || down) && (left || right);
}

interface Node { bs: number[]; pc: number; g: number; parent: number; op: number; dir: number; }

// Solve one group with push-optimal A*. allBoxesSet is every box (static for other groups).
function solveGroup(lv: Level, group: Group, startPlayer: number, active: number[], others: number[]): { moves: string; player: number; finalPos: number[] } {
  const goals = new Set(group.goals);
  const obstacles = (bs: number[]): Set<number> => {
    const s = new Set(others); for (const b of bs) s.add(b); return s;
  };
  const key = (bs: number[], pc: number): string => pc + '|' + bs.join(',');
  const nodes: Node[] = [];
  const r0 = reach(lv, obstacles(active), startPlayer);
  const pc0 = Math.min(...r0.seen);
  nodes.push({ bs: active, pc: pc0, g: 0, parent: -1, op: -1, dir: -1 });
  const best = new Map<string, number>([[key(active, pc0), 0]]);
  const heap: [number, number][] = [[heuristic(active, group.goals, lv.w), 0]]; // [f, node]
  const less = (a: [number, number], b: [number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && a[1] < b[1]);
  const pushHeap = (e: [number, number]): void => {
    let i = heap.length; heap.push(e);
    while (i > 0) { const p = (i - 1) >> 1; if (!less(heap[i]!, heap[p]!)) break; [heap[i], heap[p]] = [heap[p]!, heap[i]!]; i = p; }
  };
  const popHeap = (): [number, number] => {
    const top = heap[0]!, last = heap.pop()!;
    if (heap.length) {
      heap[0] = last;
      for (let i = 0;;) {
        let c = 2 * i + 1; if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!]; i = c;
      }
    }
    return top;
  };
  const solved = (bs: number[]): boolean => bs.every((b) => goals.has(b));
  let expanded = 0, goalNode = -1;
  while (heap.length && expanded < CAP) {
    const [, ni] = popHeap();
    const node = nodes[ni]!;
    if (best.get(key(node.bs, node.pc)) !== ni) continue;
    if (solved(node.bs)) { goalNode = ni; break; }
    expanded++;
    const boxSet = obstacles(node.bs);
    const r = reach(lv, boxSet, node.pc);
    for (let oi = 0; oi < node.bs.length; oi++) {
      const b = node.bs[oi]!;
      for (let di = 0; di < DIRS.length; di++) {
        const [dx, dy] = DIRS[di]!;
        const bx = b % lv.w, by = Math.floor(b / lv.w);
        if (bx + dx < 0 || bx + dx >= lv.w || by + dy < 0 || by + dy >= lv.h) continue;
        const dest = b + dy * lv.w + dx;
        if (lv.wall[dest] || boxSet.has(dest)) continue;
        const origin = b - (dy * lv.w + dx);
        if (!r.seen.has(origin)) continue;
        const nb = node.bs.slice(); nb[oi] = dest; nb.sort((x, y) => x - y);
        if (nb.some((x) => cornerDead(x, lv, lv.goals))) continue;
        const nr = reach(lv, obstacles(nb), b);
        const npc = Math.min(...nr.seen);
        const ng = node.g + 1;
        const k = key(nb, npc);
        const seen = best.get(k);
        if (seen !== undefined && nodes[seen]!.g <= ng) continue;
        const nn = nodes.length;
        nodes.push({ bs: nb, pc: npc, g: ng, parent: ni, op: oi, dir: di });
        best.set(k, nn);
        pushHeap([ng + heuristic(nb, group.goals, lv.w), nn]);
      }
    }
  }
  if (goalNode < 0) return { moves: '', player: startPlayer, finalPos: active };
  // Reconstruct push list (parent -> child).
  const steps: { oldBox: number; dir: number }[] = [];
  let cur = goalNode;
  while (nodes[cur]!.parent >= 0) {
    const nd = nodes[cur]!;
    const prev = nodes[nd.parent]!;
    steps.push({ oldBox: prev.bs[nd.op]!, dir: nd.dir });
    cur = nd.parent;
  }
  steps.reverse();
  let p = startPlayer; let moves = '';
  let curBoxes = active.slice();
  for (const s of steps) {
    const origin = s.oldBox - (DIRS[s.dir]![1] * lv.w + DIRS[s.dir]![0]);
    const r = reach(lv, obstacles(curBoxes), p);
    moves += walkPath(p, origin, r.parent, lv.w);
    moves += DIRS[s.dir]![2];
    curBoxes[curBoxes.indexOf(s.oldBox)] = s.oldBox + (DIRS[s.dir]![1] * lv.w + DIRS[s.dir]![0]);
    curBoxes.sort((a, b) => a - b);
    p = s.oldBox;
  }
  return { moves, player: p, finalPos: curBoxes };
}

function solve(text: string): string {
  const lv = parse(text);
  const gs = groupsOf(lv);
  let player = lv.player;
  let curPos = lv.boxes.slice();
  let out = '';
  for (const g of gs) {
    const active = g.idx.map((k) => curPos[k]!).sort((a, b) => a - b);
    const inGroup = new Set(g.idx);
    const others = curPos.filter((_, k) => !inGroup.has(k));
    const res = solveGroup(lv, g, player, active, others);
    out += res.moves;
    player = res.player;
    const sortedFinal = res.finalPos.slice().sort((a, b) => a - b);
    g.idx.slice().sort((a, b) => a - b).forEach((boxIdx, j) => { curPos[boxIdx] = sortedFinal[j]!; });
  }
  return out;
}

for (const id of readdirSync(I).sort()) {
  const level = readFileSync(join(I, id, 'level.txt'), 'utf8');
  const moves = solve(level);
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
