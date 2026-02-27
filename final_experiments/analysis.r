library(jsonlite)
library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(lme4)
library(lmerTest)
library(emmeans)

infile <- "combined_all_conditions.jsonl"

d <- stream_in(file(infile), verbose = FALSE) %>%
  as_tibble() %>%
  mutate(
    row_id   = row_number(),
    batch_id = (row_id - 1) %/% 5 + 1L
  ) %>%
  group_by(batch_id) %>%
  mutate(group_sig = str_c(sort(id), collapse = "|")) %>%
  ungroup() %>%
  mutate(
    group_id = as.integer(factor(group_sig)),
    id       = factor(id),
    mode     = factor(mode, levels = c("zeroshot", "fewshot")),
    position = as.numeric(position_in_run),
    position_f = factor(position)   # for nonlinear test
  )

# --------------------------
# Descriptive Plot
# --------------------------

ggplot(d, aes(x = position, y = pred_score, color = mode)) +
  stat_summary(fun = mean, geom = "line") +
  stat_summary(fun = mean, geom = "point") +
  labs(x = "Position in 5-essay batch",
       y = "Mean LLM score")

ggsave("position_effect_plot.png", width = 6, height = 4, dpi = 300)

# --------------------------
# MODEL 1: Linear Position Effect
# --------------------------

m_linear <- lmer(
  pred_score ~ mode * position +
    (1 | id) + (1 | group_id),
  data = d
)

summary(m_linear)
anova(m_linear)

# --------------------------
# MODEL 2: Nonlinear Position (Factor)
# --------------------------

m_factor <- lmer(
  pred_score ~ mode * position_f +
    (1 | id) + (1 | group_id),
  data = d
)

summary(m_factor)
anova(m_factor)

# --------------------------
# Estimated Marginal Means
# --------------------------

emm_position <- emmeans(m_factor, ~ position_f | mode)
pairs(emm_position)
