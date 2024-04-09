#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <cmath>
#include <queue>
#include <time.h>
#define rep(p, q) for (int p=0; p<q; p++)
#define PI 0
#define AND 1
#define NOT 2
#define STATE_WIDTH 16
#define CONNECT_SAMPLE_RATIO 0.1
using namespace std;

int countOnesInBinary(uint64_t num, int width) {
    int count = 0;
    rep (_, width) {
        if (num & 1) {
            count++;
        }
        num >>= 1;
    }
    return count;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        cout << "Failed" << endl;
        return 1;
    }
    string in_filename = argv[1];
    string out_filename = argv[2];
    
    cout << "Read File: " << in_filename << endl;
    freopen(in_filename.c_str(), "r", stdin);
    int n, m;  // number of gates
    int k_hops; 
    scanf("%d %d %d", &n, &m, &k_hops);
    cout << "Number of gates: " << n << endl;

    // Graph
    vector<int> gate_list(n);
    vector<vector<int> > fanin_list(n);
    vector<vector<int> > fanout_list(n);
    vector<int> gate_levels(n);
    vector<int> pi_list;
    int max_level = 0;

    for (int k=0; k<n; k++) {
        int type, level;
        scanf("%d %d", &type, &level);
        gate_list[k] = type;
        gate_levels[k] = level;
        if (level > max_level) {
            max_level = level;
        }
        if (type == PI) {
            pi_list.push_back(k);
        }
    }
    vector<vector<int> > level_list(max_level+1);
    for (int k=0; k<n; k++) {
        level_list[gate_levels[k]].push_back(k);
    }
    for (int k=0; k<m; k++) {
        int fanin, fanout;
        scanf("%d %d", &fanin, &fanout);
        fanin_list[fanout].push_back(fanin);
        fanout_list[fanin].push_back(fanout);
    }

    // Get hops 
    int win_level = k_hops;
    int sliding_level = 0;
    vector<vector<int> > hop_list;
    vector<int> sliding_level_list; 
    vector<int> has_hop(n); 
    while (win_level < max_level) {
        for (int k=0; k<level_list[win_level].size(); k++) {
            queue<pair<int, int> > q;
            vector<int> visited;
            q.push(pair<int, int>(level_list[win_level][k], k_hops));
            has_hop[level_list[win_level][k]] = 1;
            while (!q.empty()) {
                pair<int, int> node = q.front();
                q.pop();
                int gate = node.first;
                int hop_level = node.second;
                if (find(visited.begin(), visited.end(), gate) != visited.end()) {
                    continue;
                }
                visited.push_back(gate);
                if (hop_level == 0) {
                    continue;
                }
                for (int i=0; i<fanin_list[gate].size(); i++) {
                    q.push(pair<int, int>(fanin_list[gate][i], hop_level-1));
                }
            }
            hop_list.push_back(visited);
            sliding_level_list.push_back(sliding_level);
        }
        win_level += (k_hops - 2); 
        sliding_level += 1; 
    }

    // Add PO
    rep (k, n) {
        if (fanout_list[k].size() == 0 && has_hop[k] == 0) {
            queue<pair<int, int> > q;
            vector<int> visited;
            q.push(pair<int, int>(k, k_hops));
            has_hop[k] = 1;
            while (!q.empty()) {
                pair<int, int> node = q.front();
                q.pop();
                int gate = node.first;
                int hop_level = node.second;
                if (find(visited.begin(), visited.end(), gate) != visited.end()) {
                    continue;
                }
                visited.push_back(gate);
                if (hop_level == 0) {
                    continue;
                }
                for (int i=0; i<fanin_list[gate].size(); i++) {
                    q.push(pair<int, int>(fanin_list[gate][i], hop_level-1));
                }
            }
            hop_list.push_back(visited);
            sliding_level_list.push_back(sliding_level);
        }
    }

    // Output 
    freopen(out_filename.c_str(), "w", stdout);
    printf("%d\n", hop_list.size());
    for (int hop_idx = 0; hop_idx < hop_list.size(); hop_idx++) {
        printf("%d %d\n", hop_list[hop_idx].size(), sliding_level_list[hop_idx]);
        for (int k=0; k<hop_list[hop_idx].size(); k++) {
            printf("%d ", hop_list[hop_idx][k]);
        }
        printf("\n");
    }
}