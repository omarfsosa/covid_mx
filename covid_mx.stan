functions {
    real richards(real t, real alpha, real beta, real gamma, real S) {
        return S / (1 + gamma * exp(-beta * (t - alpha)))^(1 / gamma);
    }
    
    real d_richards(real t, real alpha, real beta, real gamma, real S) {
        real r;
        r = richards(t, alpha, beta, gamma, S);
        return (beta / gamma) * r * (1 - (r/S)^gamma);
    }
    
    real neg_binomial_2_safe_rng(real eta, real phi) {
        real gamma_rate = gamma_rng(phi, phi / eta);
        while (gamma_rate > 1073739999) {
            gamma_rate = gamma_rng(phi, phi / eta);
        }
        return poisson_rng(gamma_rate + 1e-10);
    }

}

data {
    int n_dates;
    int n_states;
    int y[n_dates, n_states];
    int<lower=0, upper=1> is_observed[n_dates, n_states];
    int pop[n_states];
    int n_future;
}

transformed data {
    real eps;
    eps = 1e-5;
}
parameters {
    // Alphas
    real mu_alpha;
    real<lower=0> sigma_alpha;
    real log_alpha[n_states];

    // Betas
    real mu_beta;
    real<lower=0> sigma_beta;
    real log_beta[n_states];

    // Gammas
    real mu_gamma;
    real<lower=0> sigma_gamma;
    real log_gamma[n_states];

    // S
    real mu_S;
    real<lower=0> sigma_S;
    real logit_S[n_states];

    // Phi
    real<lower=0> mu_phi;
    real<lower=0> sigma_phi;
    real<lower=0> phi[n_states];

    // Day of the week effect
    real<lower=-0.8, upper=0.8> dow[6]; // use day 1 as baseline, same for all state
}

transformed parameters {
    real<lower=0> alpha[n_states];
    real<lower=0> beta[n_states];
    real<lower=0> gamma[n_states];
    real<lower=0, upper=1> S[n_states];
    
    for (n in 1:n_states) {
        alpha[n] = exp(log_alpha[n]);
        beta[n] = exp(log_beta[n]);
        gamma[n] = exp(log_gamma[n]);
        S[n] = inv_logit(logit_S[n]);
    }
}

model {
    // Hyperpriors
    mu_alpha ~ normal(5, 1);
    mu_beta ~ normal(-3.5, 1);
    mu_gamma ~ normal(0, 1);
    mu_S ~ normal(-5, 2);
    mu_phi ~ normal(20, 20);

    sigma_alpha ~ student_t(5, 0, 2.5);
    sigma_beta ~ student_t(5, 0, 2.5);
    sigma_gamma ~ student_t(5, 0, 2.5);
    sigma_S ~ student_t(5, 0, 2.5);
    sigma_phi ~ normal(5, 10);
    
    
    for (n in 1:n_states) {
        // Priors
        log_alpha[n] ~ normal(mu_alpha, sigma_alpha);
        log_beta[n] ~ normal(mu_beta, sigma_beta);
        log_gamma[n] ~ normal(mu_gamma, sigma_gamma);
        logit_S[n] ~ normal(mu_S, sigma_S);
        phi[n] ~ normal(mu_phi, sigma_phi);
        
        // Likelihood
        for (t in 1:n_dates) {
            if (is_observed[t, n]) {
                real mu;
                mu =  pop[n] * d_richards(t, alpha[n], beta[n], gamma[n], S[n]);
                if (t % 7) { // for days 1, 2, 3, 4, 5 and 6
                    mu *= (1 + dow[t % 7]);
                }
                y[t, n] ~ neg_binomial_2(eps + mu, phi[n]);
            }
        }
    }
}

generated quantities {
    real y_pred[n_dates + n_future, n_states];
    real y_trend[n_dates + n_future, n_states];
    for (n in 1:n_states) {
        for (t in 1:n_dates + n_future) {
            real mu;
            mu =  pop[n] * d_richards(t, alpha[n], beta[n], gamma[n], S[n]);
            y_trend[t, n] = neg_binomial_2_safe_rng(eps + mu, phi[n]); // Trend without the day of week effect
            if (t % 7) { // for days 1, 2, 3, 4, 5 and 6
                mu *= (1 + dow[t % 7]);
            }
            y_pred[t, n] = neg_binomial_2_safe_rng(eps + mu, phi[n]);
        }
    }
}