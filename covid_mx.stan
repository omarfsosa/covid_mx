functions {
    real richards(real t, real alpha, real beta, real nu, real S) {
        return S / (1 + nu * exp(-beta * (t - alpha)))^(1 / nu);
    }
    
    real d_richards(real t, real alpha, real beta, real nu, real S) {
        real r;
        r = richards(t, alpha, beta, nu, S);
        return (beta / nu) * r * (1 - (r/S)^nu);
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
    int T;
    int N;
    int y[T, N];
    int<lower=0, upper=1> is_observed[T, N];
    int pop[N];
    int sim_T;
}

transformed data {
    real eps;
    eps = 1e-5;
}
parameters {
    vector[4] mu_theta;
    vector<lower=0>[4] sigma_theta;
    vector[4] log_theta[N];
    real<lower=0> mu_phi;
    real<lower=0> sigma_phi;
    real<lower=0> phi[N];
    real<lower=-0.8, upper=0.8> dow[6]; // use day 1 as baseline, same for all states
    
}

transformed parameters {
    real<lower=0> alpha[N];
    real<lower=0> beta[N];
    real<lower=0> nu[N];
    real<lower=0, upper=1> S[N];
    
    for (n in 1:N) {
        alpha[n] = exp(log_theta[n, 1]);
        beta[n] = exp(log_theta[n, 2]);
        nu[n] = exp(log_theta[n, 3]);
        S[n] = inv_logit(log_theta[n, 4]);
    }
}

model {
    // Hyperpriors
    mu_theta[1] ~ normal(5, 1);
    mu_theta[2] ~ normal(-3.5, 1);
    mu_theta[3] ~ normal(0, 1);
    mu_theta[4] ~ normal(-5, 2);
    for (k in 1:4) {
        sigma_theta[k] ~ cauchy(0, 2.5);
    }
    mu_phi ~ normal(20, 20);
    sigma_phi ~ normal(5, 10);
    
    
    for (n in 1:N) {
        // Priors
        for (k in 1:4) {
            log_theta[n, k] ~ normal(mu_theta[k], sigma_theta[k]);
        }
        phi[n] ~ normal(mu_phi, sigma_phi);
        
        // Likelihood
        for (t in 1:T) {
            if (is_observed[t, n]) {
                real mu;
                mu =  pop[n] * d_richards(t, alpha[n], beta[n], nu[n], S[n]);
                if (t % 7) { // for days 1, 2, 3, 4, 5 and 6
                    mu *= (1 + dow[t % 7]);
                }
                y[t, n] ~ neg_binomial_2(eps + mu, phi[n]);
            }
        }
    }
}

generated quantities {
    real y_pred[T + sim_T, N];
    real y_trend[T + sim_T, N];
    for (n in 1:N) {
        for (t in 1:T + sim_T) {
            real mu;
            mu =  pop[n] * d_richards(t, alpha[n], beta[n], nu[n], S[n]);
            y_trend[t, n] = mu;
            if (t % 7) { // for days 1, 2, 3, 4, 5 and 6
                mu *= (1 + dow[t % 7]);
            }
            y_pred[t, n] = neg_binomial_2_safe_rng(eps + mu, phi[n]);
        }
    }
}