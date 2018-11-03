#include "opencv2/opencv.hpp"
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <vector>
#include <queue>
#include <cmath>
#include <fstream>
#define rep(i, l, r) for (int i = l; i < r; i++)
#define pii pair<int, int>
#define mp make_pair
#define white 0
#define yellow 5
#define blue 1
#define green 3
#define red 2
#define orange 4
using namespace std;
using namespace cv;
typedef unsigned long long ull;

struct State {
    int c[5][5][5];
} state, finished;

const string fm[] = {"F", "F2", "F'", "R", "R2", "R'", "U", "U2", "U'",
                     "B", "B2", "B'", "L", "L2", "L'", "D", "D2", "D'", "M", "M2", "M'"};
const string whole[] = {"x", "x2", "x'", "y", "y2", "y'", "z", "z2", "z'"};
const int inv[] = {2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9, 14, 13, 12, 17, 16, 15};
const int invc[6] = {5, 3, 4, 1, 2, 0};
const char color[] = {'w', 'b', 'r', 'g', 'o', 'y'};
CvScalar cvcolor[] = {CvScalar(255, 255, 255), CvScalar(255, 0, 0), CvScalar(0, 0, 255),
                      CvScalar(0, 255, 0, 0), CvScalar(10, 180, 255), CvScalar(0, 255, 255)};
bool suf[5][5][5];
double thres = 115;
int rows, cols;
Mat frame, canny, hsv;
int side = 20;
int solution_stage = 0;
string s;
vector<string> solution[10];

int num[6][6];
int cntc;
vector<Moments> mu;
vector<vector<Point> > allface;
vector<vector<Point> > ct;
int face[3][3], refer, top, ii;
int adjusting = 1;
double facecolor[3][3];
VideoWriter writer;

void rotate0(State &state, char c) {
    State state0 = state;
    if (c == 'R') rep(i, 0, 5) rep(j, 3, 5) rep(k, 0, 5) state.c[i][j][k] = state0.c[k][j][4 - i];
    else if (c == 'L') rep(i, 0, 5) rep(j, 0, 2) rep(k, 0, 5) state.c[i][j][k] = state0.c[4 - k][j][i];
    else if (c == 'U') rep(i, 0, 5) rep(j, 0, 5) rep(k, 3, 5) state.c[i][j][k] = state0.c[4 - j][i][k];
    else if (c == 'D') rep(i, 0, 5) rep(j, 0, 5) rep(k, 0, 2) state.c[i][j][k] = state0.c[j][4 - i][k];
    else if (c == 'F') rep(i, 3, 5) rep(j, 0, 5) rep(k, 0, 5) state.c[i][j][k] = state0.c[i][4 - k][j];
    else if (c == 'B') rep(i, 0, 2) rep(j, 0, 5) rep(k, 0, 5) state.c[i][j][k] = state0.c[i][k][4 - j];
    else if (c == 'x') rep(i, 0, 5) rep(j, 0, 5) rep(k, 0, 5) state.c[i][j][k] = state0.c[k][j][4 - i];
    else if (c == 'y') rep(i, 0, 5) rep(j, 0, 5) rep(k, 0, 5) state.c[i][j][k] = state0.c[4 - j][i][k];
    else if (c == 'z') rep(i, 0, 5) rep(j, 0, 5) rep(k, 0, 5) state.c[i][j][k] = state0.c[i][4 - k][j];
    else {
        rep(i, 0, 5) rep(j, 0, 5) {
                rep(k, 3, 5) state.c[i][j][k] = state0.c[4 - j][i][k];
                rep(k, 0, 2) state.c[i][j][k] = state0.c[4 - j][i][k];
            }
    }
}

void rotate(State &state, string s) {
    rotate0(state, s[0]);
    if (s.length() == 2) {
        rotate0(state, s[0]);
        if (s[1] == '\'') rotate0(state, s[0]);
    }
}

void draw(int x, int y, int c) {
    Point pt[1][4] = {Point(x, y), Point(x, y + side), Point(x + side, y + side), Point(x + side, y)};
    const Point *ppt[1] = {pt[0]};
    int npt[] = {4};
    fillPoly(frame, ppt, npt, 1, cvcolor[c]);
}

void output() {
    vector<vector<Point> > bin[6];

    rep(i, 1, 4) rep(j, 1, 4) {
            draw(side * (3.5 + j), side * (0.5 + i), state.c[i][j][4]);
            draw(side * (0.5 + j), side * (3.5 + i), state.c[j][0][4 - i]);
            draw(side * (3.5 + j), side * (3.5 + i), state.c[4][j][4 - i]);
            draw(side * (6.5 + j), side * (3.5 + i), state.c[4 - j][4][4 - i]);
            draw(side * (9.5 + j), side * (3.5 + i), state.c[0][4 - j][4 - i]);
            draw(side * (3.5 + j), side * (6.5 + i), state.c[4 - i][j][0]);
        }
}

bool check(State state, int face[3][3]) {
    rep(i, 1, 4) rep(j, 1, 4) {
            char tmp[6];
            tmp[white] = state.c[i][j][4];
            tmp[orange] = state.c[j][0][4 - i];
            tmp[green] = state.c[4][j][4 - i];
            tmp[red] = state.c[4 - j][4][4 - i];
            tmp[blue] = state.c[0][4 - j][4 - i];
            tmp[yellow] = state.c[4 - i][j][0];
            if (tmp[face[1][1]] != face[i - 1][j - 1]) return false;
        }
    return true;
}

vector<double> red_h, orange_h;
int tmp[4][4][6];


bool check_without_ro(State state, int face[3][3]) {
    rep(i, 1, 4) rep(j, 1, 4) {
            tmp[i][j][white] = state.c[i][j][4];
            tmp[i][j][orange] = state.c[j][0][4 - i];
            tmp[i][j][green] = state.c[4][j][4 - i];
            tmp[i][j][red] = state.c[4 - j][4][4 - i];
            tmp[i][j][blue] = state.c[0][4 - j][4 - i];
            tmp[i][j][yellow] = state.c[4 - i][j][0];
        }
    rep(i, 1, 4) rep(j, 1, 4) {
            int f = face[i - 1][j - 1];
            if (f != red && f != orange && tmp[i][j][face[1][1]] != f) return false;
        }
    rep(i, 1, 4) rep(j, 1, 4) {
            if (tmp[i][j][face[1][1]] == red) red_h.push_back(facecolor[i - 1][j - 1]);
            if (tmp[i][j][face[1][1]] == orange) orange_h.push_back(facecolor[i - 1][j - 1]);
        }
    cout << 123 << endl;
    return true;
}


void rf(int n) {

    int pre[3][3];

    rep(k, 0, n) {
        rep(i, 0, 3) rep(j, 0, 3) pre[i][j] = face[i][j];
        rep(i, 0, 3) rep(j, 0, 3) face[i][j] = pre[j][2 - i];
    }
    double pre2[3][3];

    rep(k, 0, n) {
        rep(i, 0, 3) rep(j, 0, 3) pre2[i][j] = facecolor[i][j];
        rep(i, 0, 3) rep(j, 0, 3) facecolor[i][j] = pre2[j][2 - i];
    }
}

bool check(State state, int face[3][3], bool undir) {
    if (undir) {
        rep(i, 0, 4) {
            if (check(state, face)) return true;
            rf(1);
        }
        return false;
    }
    return check(state, face);
}

bool roux_check(int stage) {
    if (stage == 3) {
        rep(i, 0, 5) rep(j, 0, 5) rep(k, 0, 5)
                    if (state.c[i][j][k] != finished.c[i][j][k]) return false;
        return true;
    }
    State s = state;
    rep(i, 0, 9) rep(j, 0, 9) {
            rotate(s, whole[i]);
            rotate(s, whole[j]);
            bool flag = 1;
            rep(v, 1, 3) rep(w, 1, 4) if (s.c[0][v][w] != s.c[0][1][1]) flag = 0;
            rep(w, 2, 4) if (s.c[1][0][w] != s.c[1][0][1]) flag = 0;
            if (s.c[1][1][0] != s.c[1][2][0]) flag = 0;
            if (s.c[1][1][4] != s.c[1][2][4]) flag = 0;
            if (stage >= 1) {
                rep(v, 1, 3) rep(w, 1, 4) if (s.c[4][v][w] != s.c[4][1][1]) flag = 0;
                rep(w, 2, 4) if (s.c[3][0][w] != s.c[3][0][1]) flag = 0;
                if (s.c[1][1][0] != s.c[3][2][0]) flag = 0;
                if (s.c[1][1][4] != s.c[3][2][4]) flag = 0;
            }
            if (stage >= 2) {
                rep(u, 0, 2) rep(v, 0, 2) if (s.c[u * 2 + 1][4][v * 2 + 1] != s.c[1][4][1]) flag = 0;
                if (s.c[0][3][1] != s.c[0][3][3]) flag = 0;
                if (s.c[4][3][1] != s.c[4][3][3]) flag = 0;
                if (s.c[1][3][0] != s.c[3][3][0]) flag = 0;
            }
            if (flag) return true;
        }
    return false;
}

int compo(string a, string b) {
    int ans = 2;
    if (a.length() > 1) {if (a[1] == '\'') ans += 2; else ans++;}
    if (b.length() > 1) {if (b[1] == '\'') ans += 2; else ans++;}
    return ans % 4;
}

void findstep(State &state, int face[3][3], bool undir = 0) {
    if (check(state, face, undir)) return;
    int ans = -1;
    rep(i, 0, 21) {
        State s = state;
        rotate(s, fm[i]);
        if (check(s, face, undir)) {
            if (ans != -1) return;
            ans = i;
        }
    }
    if (ans == -1) {
        /*int ans1 = -1, ans2 = -1;
        rep(i, 0, 21) rep(j, 0, 21) {
                State s = state;
                rotate(s, fm[i]);
                rotate(s, fm[j]);
                if (check(s, face, undir)) {
                    if (ans1 != -1) return;
                    ans1 = i;
                    ans2 = j;
                }
            }
        if (ans1 == -1) return;
        rotate(state, fm[ans1]);
        rotate(state, fm[ans2]);
        solution += fm[ans1] + " " + fm[ans2] + " ";
        cout << fm[ans1] << " "  << fm[ans2] << endl;*/
        return;
    }
    rotate(state, fm[ans]);
    int size = solution[solution_stage].size();
    if (size && solution[solution_stage][size - 1][0] == fm[ans][0]) {
        int total = compo(solution[solution_stage][size - 1], fm[ans]);
        char ch = solution[solution_stage][size - 1][0];
        if (total == 0) solution[solution_stage].pop_back();
        else if (total == 1) solution[solution_stage][size - 1] = string({ch});
        else if (total == 2) solution[solution_stage][size - 1] = string({ch, '2'});
        else solution[solution_stage][size - 1] = string({ch, '\''});
    }
    else solution[solution_stage].push_back(fm[ans]);
    if (roux_check(solution_stage)) solution_stage++;
    cout << fm[ans] << endl;
}

double vcos(Point a, Point b) {
    return fabs((a.x * b.x + a.y * b.y)) / sqrt(a.x * a.x + a.y * a.y) / sqrt(b.x * b.x + b.y * b.y);
}

Point ver(Point o, Point x, Point y) {
    double a = norm(x - o), b = norm(y - o), costh = ((x - o).x * (y - o).x + (x - o).y * (y - o).y) / a / b;
    double a1 = -a * a - b * b, a0 = a * a * b * b * (1 - costh * costh);
    double r = sqrt((-a1 + sqrt(a1 * a1 - 4 * a0)) / 2);
    for (int i = -1; i <= 1; i += 2)
        for (int j = -1; j <= 1; j += 2) {
            Point3f e1((x - o).x, (x - o).y, i * sqrt(r * r - a * a)),
                    e2((y - o).x, (y - o).y, j * sqrt(r * r - b * b));
            if (fabs(e1.dot(e2)) > 1e-3) continue;
            Point3f e3 = e1.cross(e2) / r;
            if (e3.y > 0) {
                line(frame, o + Point(e3.x, e3.y), o, Scalar(0, 0, 255), 3, CV_AA);
                return o + Point(e3.x, e3.y);
            }
        }

}

int recog(double h, double s) {
    if (s < 60) return white;
    if (h < 20) return blue;
    if (h < 60) return green;
    if (h < 100) return yellow;
    if (h < thres) return orange;
    return red;
}

void init() {
    rep(i, 0, 5)rep(j, 0, 5)rep(k, 0, 5) finished.c[i][j][k] = -1;
    rep(i, 1, 4)rep(j, 1, 4) {
            finished.c[i][j][4] = white;
            finished.c[i][j][0] = yellow;
            finished.c[i][4][j] = red;
            finished.c[i][0][j] = orange;
            finished.c[4][i][j] = green;
            finished.c[0][i][j] = blue;
        }
    rep(i, 0, 5)rep(j, 0, 5)num[i][j] = -1;
    state = finished;
    int length = 20, x, lastx = -3;

    rep (i, 0, length) {
        do {
            x = rand() % 18;
        } while (x / 3 == lastx / 3);
        lastx = x;
        s += " " + fm[x];
        rotate(state, fm[x]);
    }
    cout << s;
    num[white][green] = num[yellow][blue] = num[blue][yellow] = num[green][yellow] = num[red][yellow] = num[orange][yellow] = 0;
    num[white][red] = num[yellow][red] = num[blue][orange] = num[green][red] = num[red][blue] = num[orange][green] = 1;
    num[white][blue] = num[yellow][green] = num[blue][white] = num[green][white] = num[red][white] = num[orange][white] = 2;
    num[white][orange] = num[yellow][orange] = num[blue][red] = num[green][orange] = num[red][green] = num[orange][blue] = 3;
}


void getface(Mat &frame) {
    mu.clear();
    allface.clear();
    ct.clear();
    Mat element = getStructuringElement(0, Size(2 + 1, 2 + 1), Point(1, 1));
    GaussianBlur(frame, frame, Size(3, 3), 0, 0);
    Canny(frame, canny, 50, 150);
    dilate(canny, canny, element);

    vector<vector<Point> > contours;
    vector<int> faces;

    findContours(canny, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
    rep (i, 0, contours.size()) if (contours[i].size() > 100) ct.push_back(contours[i]);
    cntc = ct.size();
    mu = vector<Moments>(cntc);
    bool flag = 0;
    rep (i, 0, cntc) {
        mu[i] = moments(ct[i]);
        double hu[7];
        HuMoments(mu[i], hu);
        if (fabs(hu[0] - 0.166) < 0.02) {
            faces.push_back(i);
            //drawContours(frame, ct, i, 255, 5);
        }
    }
    int cntface = faces.size();
    rep (i, 0, cntface)
        rep(j, 0, cntface) if (i != j)
                rep(k, j + 1, cntface) if (k != i) {
                        int o = faces[i], a = faces[j], b = faces[k];
                        double ox = mu[o].m10 / mu[o].m00, oy = mu[o].m01 / mu[o].m00;
                        double ax = mu[a].m10 / mu[a].m00, ay = mu[a].m01 / mu[a].m00;
                        double bx = mu[b].m10 / mu[b].m00, by = mu[b].m01 / mu[b].m00;
                        double a00 = ax - ox, a01 = ay - oy, a10 = bx - ox, a11 = by - oy;
                        double sum = 0;
                        vector<vector<Point>> poly(1);
                        approxPolyDP(ct[o], poly[0], 10, 1);
                        //drawContours(frame, poly, 0, CvScalar(0, 255, 0), 5);
                        if (poly[0].size() != 4) continue;

                        double cri = vcos(poly[0][1] - poly[0][0], Point(a00, a01)) +
                                     vcos(poly[0][2] - poly[0][1], Point(a10, a11)) +
                                     vcos(poly[0][3] - poly[0][2], Point(a00, a01)) +
                                     vcos(poly[0][0] - poly[0][3], Point(a10, a11));
                        if (fabs(cri) > 0.1 && fabs(cri - 4) > 0.1) continue;


                        double det = a00 * a11 - a10 * a01;
                        double b00 = a11 / det, b01 = -a01 / det, b10 = -a10 / det, b11 = a00 / det;
                        //vector<pii > cord;
                        int mina = 999, maxa = -999, minb = 999, maxb = -999;
                        rep(l, 0, cntface) if (l != i && l != j && l != k) {
                                int ll = faces[l];
                                double lx = mu[ll].m10 / mu[ll].m00, ly = mu[ll].m01 / mu[ll].m00;
                                double A = b00 * (lx - ox) + b10 * (ly - oy);
                                double B = b01 * (lx - ox) + b11 * (ly - oy);
                                if (fabs(round(A) - A) < 0.1 && fabs(round(B) - B) < 0.1) {
                                    int ra = cvRound(A), rb = cvRound(B);
                                    mina = std::min(mina, ra);
                                    maxa = std::max(maxa, ra);
                                    minb = std::min(minb, rb);
                                    maxb = std::max(maxb, rb);
                                    //cord.push_back(mp(ra, rb));
                                }
                            }
                        if (maxa - mina == 2 && maxb - minb == 2) {
                            vector<Point> pt;
                            double mg = 0.5;
                            double aa[2] = {mina - mg, maxa + mg}, bb[2] = {minb - mg, maxb + mg};
                            rep (u, 0, 2) rep(v, 0, 2)pt.push_back(Point(ox + aa[u] * (ax - ox) + bb[v] * (bx - ox),
                                                                         oy + aa[u] * (ay - oy) +
                                                                         bb[v] * (by - oy)));
                            allface.push_back(pt);
                        }
                    }
}


void readface() {
    vector<Point> pt = allface[0];
    Point u = pt[1] - pt[0], v = pt[2] - pt[0];
    int sgn = CV_SIGN(u.x * v.y - u.y * v.x);
    if (sgn == 1) swap(pt[1], pt[2]);
    int cnt = 1;
    rep(i, 1, allface.size()) {
        vector<Point> ppt = allface[i];
        u = ppt[1] - ppt[0], v = ppt[2] - ppt[0];
        if (CV_SIGN(u.x * v.y - u.y * v.x) == 1)
            swap(allface[i][1], allface[i][2]);
        bool flag = 0;

        rep(k, 0, 4) {
            if (norm(allface[i][0] - allface[0][0]) < 20) {
                flag = 1;
                cnt++;
                break;
            }
            Point tmp = allface[i][0];
            allface[i][0] = allface[i][2];
            allface[i][2] = allface[i][3];
            allface[i][3] = allface[i][1];
            allface[i][1] = tmp;
        }
        if (flag) rep(j, 0, 4) pt[j] += allface[i][j];
    }
    rep(i, 0, 4) pt[i] /= cnt;
    line(frame, pt[0], pt[1], Scalar(0, 0, 255), 3, CV_AA);
    line(frame, pt[0], pt[2], Scalar(0, 0, 255), 3, CV_AA);
    line(frame, pt[3], pt[2], Scalar(0, 0, 255), 3, CV_AA);
    line(frame, pt[3], pt[1], Scalar(0, 0, 255), 3, CV_AA);

    rep(i, 0, 3) rep(j, 0, 3) {
            Point center =
                    pt[0] + (pt[1] - pt[0]) * (0.167 + 0.333 * i) + (pt[2] - pt[0]) * (0.167 + 0.333 * j);
            double sumh = 0, sums = 0;
            int pad = 10;
            rep(p, center.x - pad, center.x + pad)rep(q, center.y - pad, center.y + pad)if (p > 0 && q > 0 &&
                                                                                            q < rows &&
                                                                                            p < cols) {
                        sumh += hsv.data[q * cols * 3 + p * 3];
                        sums += hsv.data[q * cols * 3 + p * 3 + 1];
                    }
            sumh /= pad * pad * 4;
            sums /= pad * pad * 4;
            //cout << sumh << " " << sums << endl;
            face[i][j] = recog(sumh, sums);
            facecolor[i][j] = sumh;
        }

    if (adjusting) return;
    swap(pt[2], pt[3]);

    refer = -1;
    top = 0;
    Point footpt[4];
    rep(i, 1, 4) if (pt[i].y < pt[top].y) top = i;
    rep(i, 0, 4) if (i != top) footpt[i] = ver(pt[i], pt[(i + 1) % 4], pt[(i + 3) % 4]);
    rep(i, 0, 1) {
        int p = (top + 1 + i) % 4, q = (top + 2 + i) % 4;
        if (fabs(pt[p].x - pt[q].x) > 40) {
            Point center = (pt[p] + pt[q] + footpt[p] + footpt[q]) / 4;
            //line(frame, center, center, Scalar(0, 0, 255), 10, CV_AA);
            rep(j, 0, cntc) {
                double x = mu[j].m10 / mu[j].m00, y = mu[j].m01 / mu[j].m00;
                if (norm(Point(x, y) - center) < 15) {
                    double hu[7];
                    HuMoments(mu[j], hu);
                    if (fabs(hu[0] - 0.166) > 0.05) continue;
                    drawContours(frame, ct, j, CvScalar(0, 0, 255), 5);
                    int pad = 5;
                    double sumh = 0, sums = 0;
                    rep(p, center.x - pad, center.x + pad)rep(q, center.y - pad, center.y + pad) {
                            sumh += hsv.data[q * cols * 3 + p * 3];
                            sums += hsv.data[q * cols * 3 + p * 3 + 1];
                        }
                    sumh /= pad * pad * 4;
                    sums /= pad * pad * 4;
                    refer = recog(sumh - 1, sums);
                    ii = i;
                    cout << color[refer] << endl;
                    break;
                }
            }
        }
    }
}

void on_MouseHandle(int event, int x, int y, int flags, void *param) {
    if (event == EVENT_LBUTTONUP) {
        adjusting = 0;

    }

}


int main() {
    //string name;
    //cout << "Enter your name" << endl;
    //cin >> name;
    int seed = 19980514;
    srand(seed);
    init();
    VideoCapture cap(0);//"../feliks.mp4");
    if (!cap.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }
    namedWindow("wnd");
    setMouseCallback("wnd", on_MouseHandle);
    //rep(i, 0, 100) cap >> frame;
    int framecnt = 0;
    while (1) {
        framecnt++;
        cap >> frame;
        if (frame.empty()) break;
        cvtColor(frame, hsv, CV_RGB2HSV);

        if (rows == 0) {
            rows = frame.rows;
            cols = frame.cols;
            writer = VideoWriter("Demo0.avi", CV_FOURCC('M', 'J', 'P', 'G'), 8.0, Size(cols, rows));
        }
        getface(frame);
        if (!allface.empty()) {
            readface();
            if (adjusting) {
                rep (i, 0, 3) {
                    rep(j, 0, 3) cout << facecolor[i][j] << " ";
                    cout << endl;
                }
                rep (i, 0, 4) {
                    check_without_ro(state, face);
                    rf(1);
                }
                if (face[1][1] == red) {
                    face[1][1] = orange;
                    rep (i, 0, 4) {
                        check_without_ro(state, face);
                        rf(1);
                    }
                } else if (face[1][1] == orange) {
                    face[1][1] = red;
                    rep (i, 0, 4) {
                        check_without_ro(state, face);
                        rf(1);
                    }
                }
            } else {
                bool flag = 1;
                if (refer != -1) {
                    if (num[face[1][1]][refer] == -1) flag = 0;
                    rf(4 - (top + ii) + num[face[1][1]][refer]);
                }
                if (refer != -1 && flag) {

                    findstep(state, face);
                } else findstep(state, face, 1);
            }

        }

        output();
        int baseline;
        Size text_size = getTextSize(s, CV_FONT_NORMAL, 1, 1, &baseline);
        rectangle(frame, Point(200, 50),
                  Point(200 + text_size.width, 50 - text_size.height), CvScalar(255, 255, 255), CV_FILLED);
        putText(frame, s, Point(200, 50), CV_FONT_NORMAL, 1, Scalar(0, 0, 0), 1, 8, 0);
        string ind[4] = {"The first bridge: ", "The second bridge: ", "CMLL:", "Last six edges: "};
        rep(i, 0, 4) {
            string sol;
            rep(j, 0, solution[i].size()) sol += solution[i][j] + " ";
            putText(frame, ind[i] + sol, Point(50, 400 + i * 50), CV_FONT_NORMAL, 1, Scalar(0, 0, 255), 1, 8,
                    0);
        }
        int steps = 0;
        rep (i, 0, 4) steps += solution[i].size();
        putText(frame, to_string(steps) + " Steps in total", Point(50, 650), CV_FONT_NORMAL, 1, Scalar(0, 0, 255), 1, 8, 0);

        if (adjusting) {
            double minh = 999, maxh = 0;
            rep (i, 0, red_h.size()) minh = min(minh, red_h[i]);
            rep (i, 0, orange_h.size()) maxh = max(maxh, orange_h[i]);
            thres = (minh + maxh) / 2;
            putText(frame, to_string(minh) + " " + to_string(maxh), Point(50, 350), CV_FONT_NORMAL, 1, Scalar(0, 0, 255), 1, 8, 0);
            string adj = "Adjusting..Click to start";
            putText(frame, adj, Point(50, 600), CV_FONT_NORMAL, 1, Scalar(0, 0, 255), 1, 8, 0);
        }
        else {
            putText(frame, "Press Q to Quit", Point(50, 600), CV_FONT_NORMAL, 1, Scalar(0, 0, 255), 1, 8, 0);
        }
        imshow("wnd", frame);
        writer << frame;
        auto c = (char) waitKey(1);
        if (c == 'q') {
            break;
        }

    }

    writer.release();
    cap.release();
    destroyAllWindows();
}