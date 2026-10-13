// bprime_symbolic.cpp

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace bprime {

const double P_MIN = 100.0, P_MAX = 150100.0;
const double T_MIN = 250.0, T_MAX = 2500.0;

// Scaled inputs used by every fit: x0 = log10(P) and x1 = T, both mapped onto [-1, 1].
inline void scaled_inputs(double P, double T, double &x0, double &x1)
{
    P = std::min(std::max(P, P_MIN), P_MAX);
    T = std::min(std::max(T, T_MIN), T_MAX);
    x0 = 2.0 * (std::log10(P) - std::log10(P_MIN)) / (std::log10(P_MAX) - std::log10(P_MIN)) - 1.0;
    x1 = 2.0 * (T - T_MIN) / (T_MAX - T_MIN) - 1.0;
}

inline double horner(const std::vector<double> &c, double x) // c[0] + c[1] x + c[2] x^2 + ...
{
    double s = 0.0;
    for (std::size_t k = c.size(); k-- > 0;)
        s = s * x + c[k];
    return s;
}

// ------------------------------------------------------------------------------------------------
// Species fits. Each fills y[1..] with the dominant species; y[0] = Y_N2 is set by the caller.
// ------------------------------------------------------------------------------------------------

// Carbon: one Clausius-Clapeyron-like coordinate xi = log10 P + 8966.5/T - 4 (written in x0, x1)
// drives both species; Y_CO2 is the exact affine image of Y_CO (a one-parameter composition).
inline void carbon_species(double x0, double x1, double *y)
{
    const double T = 1125.0 * x1 + 1375.0;
    const double t1 = std::tanh(1.03092289536785 * x0 - 6.65853433119674 + 5820.32501284749 / T);
    const double t2 = std::tanh(1.03135782835988 * x0 - 6.658496427579 + 5822.78053244433 / T);
    y[1] = -38.0082831184339 * t1 + 37.8348884086489 * t2 + 0.17369192736926;   // CO
    y[2] = 32.2600799151313 * t1 - 32.1129086478291 * t2 + 0.147166437729977;   // CO2
}

// SiC: two independent fronts (C and Si leave the wall with independent extents).
// tanh(c / x1^4) is continuous through x1 = 0 (T = 1375 K) only because IEEE-754 gives c/0 = +inf and
// tanh(+inf) = 1; compile without -ffast-math and without floating-point exception trapping.
inline void sic_species(double x0, double x1, double *y)
{
    const double x1_2 = x1 * x1, x1_4 = x1_2 * x1_2;
    const double L = std::log(std::pow(127.450896354941 * x1 - 143.190140419534, 2));
    y[1] = (0.104485664795769 * x1_4 - 0.157048330600228 * std::tanh(x1 * L * L - 46.3068390559413))
           * std::tanh(0.465969431834478 * std::exp(-4.0 * x0 + 20.5109436863665 * x1));          // CO
    double s = x1 + 0.5731647;
    s *= s; s *= s; s *= s; s *= s;                                                        // (x1 + 0.573)^16
    y[2] = 0.17561531 * std::tanh(0.5514107 / x1_4) * std::tanh(s * std::exp(-6.0 * x0 + 8.0 * x1)); // SiO
}

// ZrC: CO from the oxidation front, ZrO from the onset of vaporisation at low pressure.
inline void zrc_species(double x0, double x1, double *y)
{
    double b = std::tanh(x1) + 0.61724526;
    b *= b; b *= b; b *= b; b *= b; b *= b; b *= b;                                        // b^64
    y[1] = 0.1506072 * std::pow(std::tanh(30695173.9538546 * b * std::exp(-16.0 * x0)), 0.25);         // CO
    const double x1_2 = x1 * x1, x1_4 = x1_2 * x1_2, x1_8 = x1_4 * x1_4;
    double u = -x1 - 0.873132; u *= u; u *= u;                                             // ^4
    double v = x1 * (1.0 - x0); v *= v; v *= v; v *= v;                                    // ^8
    y[2] = 0.18171257 * std::tanh(0.000596831900295455 * x1_8 * u * v);                    // ZrO
}

// ------------------------------------------------------------------------------------------------
// Surrogate = species fits + species enthalpies (polynomials in x1) + the B'c relation.
// ------------------------------------------------------------------------------------------------

struct Surrogate
{
    std::string name, table;                 // material, table file name
    std::vector<std::string> species;        // "N2" first, then the fitted species
    void (*fit)(double, double, double *);
    std::vector<std::vector<double> > h;     // h_i(x1) [J/kg], same order as species
    double z_n_edge;                         // nitrogen mass fraction of the edge air
    std::vector<double> bc_lin;              // linear term on the fitted species (not N2)
    double bc_const;

    struct State { std::vector<double> y; double hw, bc; };

    State eval(double P, double T) const
    {
        double x0, x1;
        scaled_inputs(P, T, x0, x1);
        State s;
        s.y.assign(species.size(), 0.0);
        fit(x0, x1, s.y.data());
        double y_n2 = 1.0;
        for (std::size_t i = 1; i < species.size(); i++)
            y_n2 -= s.y[i];
        s.y[0] = y_n2;
        s.hw = 0.0;
        for (std::size_t i = 0; i < species.size(); i++)
            s.hw += s.y[i] * horner(h[i], x1);
        // nitrogen is an inert tracer: Z_N,wall (1 + B') = Z_N,edge
        s.bc = z_n_edge / y_n2 - 1.0 + bc_const;
        for (std::size_t i = 1; i < species.size(); i++)
            s.bc += bc_lin[i - 1] * s.y[i];
        return s;
    }
};

// NASA-9 enthalpies [J/kg] as polynomials in x1: degree 3 for Carbon/SiC, degree 6 for ZrC.
const std::vector<double> H3_N2 = {1216157.5310546875, 1379769.7623046874, 86179.5228515625, -30756.253710937497};
const std::vector<double> H3_CO = {-2716350.710546875, 1396933.076953125, 84440.69648437497, -34721.187890625};
const std::vector<double> H3_CO2 = {-7707137.970507813, 1477241.4076171876, 122160.86191406252, -70373.9865234375};
const std::vector<double> H3_SIO = {-2325075.3564453125, 936892.6154296875, 36148.354101562494, -24502.8005859375};
const std::vector<double> H6_N2 = {1216359.866487903, 1383450.910131708, 80528.5138339391, -47843.548174409276,
                                   19308.53999632294, 15307.976859495446, -15220.067182097664};
const std::vector<double> H6_CO = {-2715669.989693244, 1399968.8694310393, 74106.41540552366, -48887.93355254797,
                                   25458.115906436673, 12697.234039558076, -16074.051706861097};
const std::vector<double> H6_ZRO = {1226385.5731566497, 435938.43388983986, -25151.803713827365, 47363.58075544638,
                                    -28739.970272417853, -39978.53402779975, 38838.53257656222};

std::vector<Surrogate> surrogates()
{
    std::vector<Surrogate> m(3);
    m[0] = {"Carbon", "Carbon_Bprime.dat", {"N2", "CO", "CO2"}, carbon_species, {H3_N2, H3_CO, H3_CO2},
            0.767082, {0.0001928910401, -0.0001509663169}, 4.328980563e-05};
    // SiC table (Munafo): its Y_ columns are MOLE fractions, so Z_N,e / X_N2 - 1 misses B'c by the
    // mole-to-mass conversion; the 0.734 X_SiO term fits that conversion (it is not a condensed phase).
    m[1] = {"SiC", "SiC_Bprime.dat", {"N2", "CO", "SiO"}, sic_species, {H3_N2, H3_CO, H3_SIO},
            0.767082, {-0.03123388148, 0.7339526229}, 0.0002337201863};
    m[2] = {"ZrC", "ZrC_Bprime.dat", {"N2", "CO", "ZrO"}, zrc_species, {H6_N2, H6_CO, H6_ZRO},
            0.7670824885424203, {0.0, 0.0}, 0.0};
    return m;
}

} // namespace bprime

// ------------------------------------------------------------------------------------------------
// Reading a table:  p  T  B'c  Y_<species>...  h  (header line names the columns; '#' lines skipped)
// ------------------------------------------------------------------------------------------------

struct Table
{
    std::vector<std::string> cols;
    std::vector<std::vector<double> > rows;
    int col(const std::string &name) const
    {
        for (std::size_t j = 0; j < cols.size(); j++)
            if (cols[j] == name) return int(j);
        return -1;
    }
};

bool is_number(const std::string &s)
{
    char *end = nullptr;
    std::strtod(s.c_str(), &end);
    return end != s.c_str() && *end == '\0';
}

bool read_table(const std::string &path, Table &t)
{
    std::ifstream in(path.c_str());
    if (!in) return false;
    std::string line;
    while (std::getline(in, line))
    {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::vector<std::string> tok;
        for (std::string w; ss >> w;) tok.push_back(w);
        if (tok.size() < 5) continue; // species count, grid size
        if (!tok[0].empty() && tok[0].compare(0, 2, "//") == 0)
        {
            if (tok[0] == "//") tok.erase(tok.begin()); else tok[0] = tok[0].substr(2);
        }
        if (!is_number(tok[0])) { t.cols = tok; continue; }
        std::vector<double> r;
        for (std::size_t j = 0; j < tok.size(); j++) r.push_back(std::atof(tok[j].c_str()));
        t.rows.push_back(r);
    }
    return !t.rows.empty() && t.cols.size() == t.rows[0].size();
}

// ------------------------------------------------------------------------------------------------
// Error statistics
// ------------------------------------------------------------------------------------------------

struct Err
{
    double sum2 = 0, max_abs = 0, sum_rel = 0, lo = 1e300, hi = -1e300;
    long n = 0, n_rel = 0;
    void add(double truth, double model, double rel_floor)
    {
        const double e = model - truth;
        sum2 += e * e; n++;
        max_abs = std::max(max_abs, std::fabs(e));
        lo = std::min(lo, truth); hi = std::max(hi, truth);
        if (std::fabs(truth) >= rel_floor) { sum_rel += std::fabs(e / truth); n_rel++; }
    }
    double rms() const { return std::sqrt(sum2 / n); }
    double rms_of_range() const { return 100.0 * rms() / (hi - lo); }
    double mean_rel() const { return n_rel ? 100.0 * sum_rel / n_rel : 0.0; }
};

void compare(const bprime::Surrogate &m, const std::string &dir)
{
    Table t;
    const std::string path = dir + "/" + m.table;
    if (!read_table(path, t))
    {
        std::printf("\n%s: cannot read %s, skipped\n", m.name.c_str(), path.c_str());
        return;
    }
    const int cB = 2, cH = int(t.cols.size()) - 1;
    std::vector<int> cY;
    for (std::size_t i = 0; i < m.species.size(); i++)
    {
        cY.push_back(t.col("Y_" + m.species[i]));
        if (cY.back() < 0) { std::printf("%s: no column Y_%s\n", path.c_str(), m.species[i].c_str()); return; }
    }

    std::printf("\n================================================================================\n");
    std::printf("%s   (%s, %zu states)\n", m.name.c_str(), path.c_str(), t.rows.size());
    std::printf("================================================================================\n");

    // 1. sampled profiles: table vs surrogate at three pressures, every 250 K
    std::printf("%9s %6s | %9s %9s | %9s %9s |", "P[Pa]", "T[K]", "B'c tab", "model", "h_w tab", "model");
    for (std::size_t i = 1; i < m.species.size(); i++)
        std::printf(" %7s %7s |", ("Y_" + m.species[i]).c_str(), "model");
    std::printf("\n");
    const double P_show[3] = {100.0, 10100.0, 150100.0};
    for (double Ps : P_show)
        for (const auto &r : t.rows)
        {
            if (std::fabs(r[0] - Ps) > 0.5 || std::fmod(r[1] - 250.0, 250.0) != 0.0) continue;
            const auto s = m.eval(r[0], r[1]);
            std::printf("%9.0f %6.0f | %9.5f %9.5f | %9.4g %9.4g |", r[0], r[1], r[cB], s.bc, r[cH], s.hw);
            for (std::size_t i = 1; i < m.species.size(); i++)
                std::printf(" %7.4f %7.4f |", r[cY[i]], s.y[i]);
            std::printf("\n");
        }

    // 2. error over every table state
    Err eB, eH;
    std::vector<Err> eY(m.species.size());
    double bmax = 0;
    for (const auto &r : t.rows) bmax = std::max(bmax, std::fabs(r[cB]));
    for (const auto &r : t.rows)
    {
        const auto s = m.eval(r[0], r[1]);
        eB.add(r[cB], s.bc, 0.05 * bmax); // relative error only where |B'c| >= 5% of its maximum
        eH.add(r[cH], s.hw, 1.0e4);        // ... and where |h_w| >= 10 kJ/kg (h_w crosses zero)
        for (std::size_t i = 0; i < m.species.size(); i++) eY[i].add(r[cY[i]], s.y[i], 1e300);
    }
    std::printf("\nerror over all %zu table states\n", t.rows.size());
    std::printf("  %-7s  RMS %-11.4g  RMS/range %6.3f%%  max abs %-10.4g  mean rel %6.3f%%\n", "B'c", eB.rms(),
                eB.rms_of_range(), eB.max_abs, eB.mean_rel());
    std::printf("  %-7s  RMS %-11.4g  RMS/range %6.3f%%  max abs %-10.4g  mean rel %6.3f%%   [J/kg]\n", "h_w",
                eH.rms(), eH.rms_of_range(), eH.max_abs, eH.mean_rel());
    for (std::size_t i = 0; i < m.species.size(); i++)
        std::printf("  %-7s  RMS %-11.4g  RMS/range %6.3f%%  max abs %-10.4g\n", ("Y_" + m.species[i]).c_str(),
                    eY[i].rms(), eY[i].rms_of_range(), eY[i].max_abs);

    // 3. dense sampling between the table nodes: finite, physical, smooth
    const int NP = 301, NT = 2251; // log-spaced P, 1 K in T
    double ymin = 1e300, ymax = -1e300, jump = 0, prev = 0;
    bool finite = true;
    for (int a = 0; a < NP; a++)
    {
        const double P = bprime::P_MIN * std::pow(bprime::P_MAX / bprime::P_MIN, a / double(NP - 1));
        for (int b = 0; b < NT; b++)
        {
            const double T = bprime::T_MIN + (bprime::T_MAX - bprime::T_MIN) * b / double(NT - 1);
            const auto s = m.eval(P, T);
            finite = finite && std::isfinite(s.hw) && std::isfinite(s.bc);
            for (double y : s.y) { ymin = std::min(ymin, y); ymax = std::max(ymax, y); }
            if (b > 0) jump = std::max(jump, std::fabs(s.hw - prev));
            prev = s.hw;
        }
    }
    std::printf("dense sampling (%d x %d states, 1 K apart): all finite = %s, Y in [%.3g, %.4f], "
                "largest h_w step over 1 K = %.4g J/kg\n", NP, NT, finite ? "yes" : "NO", ymin, ymax, jump);
}

int main(int argc, char **argv)
{
    const auto models = bprime::surrogates();

    if (argc == 5 && std::string(argv[1]) == "--eval")
    {
        for (const auto &m : models)
            if (m.name == argv[2])
            {
                const double P = std::atof(argv[3]), T = std::atof(argv[4]);
                const auto s = m.eval(P, T);
                std::printf("%s  P = %g Pa  T = %g K\n", m.name.c_str(), P, T);
                for (std::size_t i = 0; i < m.species.size(); i++)
                    std::printf("  Y_%-4s = %.10g\n", m.species[i].c_str(), s.y[i]);
                std::printf("  h_w    = %.10g J/kg\n  B'c    = %.10g\n", s.hw, s.bc);
                return 0;
            }
        std::fprintf(stderr, "unknown material %s (Carbon, SiC, ZrC)\n", argv[2]);
        return 1;
    }

    const std::string dir = (argc > 1) ? argv[1] : "data";
    for (const auto &m : models)
        compare(m, dir);
    return 0;
}
