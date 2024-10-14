#include <RcppArmadillo.h>

#include <iostream>

// [[Rcpp::plugins(openmp)]]

#pragma omp declare reduction(+ : arma::mat : omp_out += omp_in) \
    initializer(omp_priv = omp_orig)

double logdens_g_minimal(const arma::vec hx, const arma::vec mu_g,
                         const arma::mat sigma_inv) {
    arma::vec qx = hx - mu_g;
    arma::vec res = (qx.t() * sigma_inv * qx);
    double num = -0.5 * res(0);
    return num;
}

double density_g_minimal(const arma::vec hx, const arma::vec mu_g,
                         const arma::mat sigma_inv) {
    arma::vec qx = hx - mu_g;
    arma::vec res = (qx.t() * sigma_inv * qx);
    double num = std::exp(-0.5 * res(0));
    return num;
}

double density_minimal(const arma::vec hx, const arma::vec pi_all,
                       const arma::mat mu_all, const arma::mat sigma_inv,
                       const int G) {
    double result, piece;
    result = 0;
    for (int g = 0; g < G; g++) {
        piece = pi_all[g] * density_g_minimal(hx, mu_all.row(g).t(), sigma_inv);
        result += piece;
    }
    return result;
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
Rcpp::List em_algorithm_cpp(arma::cube data, arma::cube h, arma::cube w,
                            arma::vec pi_all, arma::mat mu_all, arma::mat sigma,
                            arma::uword NUM_STEPS) {
    Rcpp::List answer;
    arma::uword G = mu_all.n_rows;  // 16
    arma::uword d = mu_all.n_cols;  // 3
    // h is given already
    arma::uword N = data.n_rows * data.n_cols;

    arma::mat mu_old(mu_all);
    arma::vec pi_old(pi_all);
    arma::mat sigma_old(sigma);
    //
    arma::mat mu_next(mu_all);
    arma::vec pi_next(pi_all);
    arma::mat sigma_next(sigma);
    //
    arma::mat U(3, 3);
    arma::uword k;

    mu_next.fill(0.0);
    pi_next.fill(0.0);
    sigma_next.fill(0.0);

    for (k = 0; k < NUM_STEPS; k++) {
        // calculate w
        arma::mat soi = sigma_old.i();
#pragma omp parallel for collapse(2)
        for (arma::uword i = 0; i < data.n_rows; i++) {
            for (arma::uword j = 0; j < data.n_cols; j++) {
                arma::vec hxi = h.tube(i, j);
                double den = density_minimal(hxi, pi_old, mu_old, soi, G);
                double lden = std::log(den);
                for (arma::uword g = 0; g < G; g++) {
                    double dmg =
                        std::log(pi_old[g]) +
                        logdens_g_minimal(hxi, mu_old.row(g).t(), soi) - lden;
                    w(i, j, g) = std::exp(dmg);
                }
            }
        }

        // calculate pi^(k+1) and mu^(k+1)
#pragma omp parallel for
        for (arma::uword g = 0; g < G; g++) {
            double wd = arma::accu(w.slice(g));
            pi_next[g] = wd / N;
            mu_next(g, 0) = arma::accu(w.slice(g) % h.slice(0)) / wd;
            mu_next(g, 1) = arma::accu(w.slice(g) % h.slice(1)) / wd;
            mu_next(g, 2) = arma::accu(w.slice(g) % h.slice(2)) / wd;
        }

        // calculate sigma^(k+1)
        U.fill(0.0);
#pragma omp parallel for collapse(2) reduction(+ : U)
        for (arma::uword i = 0; i < data.n_rows; i++) {
            for (arma::uword j = 0; j < data.n_cols; j++) {
                arma::vec h_id = h.tube(i, j);
                arma::vec h_dmu_g(3);
                arma::mat utmp(3, 3);
                utmp.fill(0.0);
                for (arma::uword g = 0; g < G; g++) {
                    h_dmu_g = h_id - mu_old.row(g).t();
                    auto sub = (w(i, j, g) * (h_dmu_g * h_dmu_g.t()));
                    utmp += sub / N;
                }
                U += utmp;
            }
        }
        sigma_next = U;

        // update old
        pi_old = pi_next;
        mu_old = mu_next;
        sigma_old = sigma_next;
    }

    answer = Rcpp::List::create(Rcpp::Named("mu") = mu_old,
                                Rcpp::Named("pi") = pi_old,
                                Rcpp::Named("sigma") = sigma_old);
    return answer;
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::umat get_cids_cpp(arma::cube data, arma::cube h, arma::mat mu) {
    arma::umat answer(data.n_rows, data.n_cols);
    arma::uword G = mu.n_rows;  // 16
    // h is given already
    answer.fill(0.0);

#pragma omp parallel for collapse(2)
    for (arma::uword i = 0; i < data.n_rows; i++) {
        for (arma::uword j = 0; j < data.n_cols; j++) {
            arma::vec hxi = h.tube(i, j);
            double min_dist = 100000.0;
            arma::uword dist_id = 0;
            for (arma::uword g = 0; g < G; g++) {
                double dmg = arma::norm(hxi - mu.row(g).t());
                if (dmg < min_dist) {
                    min_dist = dmg;
                    dist_id = g;
                }
            }
            answer(i, j) = dist_id;
        }
    }

    return answer;
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::cube get_qimg_cpp(arma::cube data, arma::umat cids, arma::mat mu) {
    arma::cube answer(data.n_rows, data.n_cols, data.n_slices);
    answer.fill(0.0);

#pragma omp parallel for collapse(2)
    for (arma::uword i = 0; i < data.n_rows; i++) {
        for (arma::uword j = 0; j < data.n_cols; j++) {
            answer.tube(i, j) = mu.row(cids(i, j)).t();
        }
    }

    return answer;
}
