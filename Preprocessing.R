library(tidyverse)
library(readxl)
library(openxlsx)


# List all Excel files in the folder
excel_files <- list.files(path = "Data/Training folder", pattern = "\\.xlsx$", full.names = TRUE)

# Create a new workbook
wb <- createWorkbook()

# Loop through each file and add as a new sheet
for (file in excel_files) {
  sheet_name <- tools::file_path_sans_ext(basename(file))
  data <- read_excel(file)
  addWorksheet(wb, sheet_name)
  writeData(wb, sheet_name, data)
}

# Save the merged workbook
saveWorkbook(wb, file = file.path("Data", "merged_output.xlsx"), overwrite = TRUE)

# Baseline evaluation

data <- read_excel("Data/merged_output.xlsx", sheet = 1)

# 필요한 패키지 설치 및 로드
# install.packages(c("readxl", "dplyr", "tidyr", "stringr"))
library(readxl)
library(dplyr)
library(tidyr)
library(stringr)


df_raw <- data
# 1) Excel 불러오기 (헤더 바로 위 빈 행을 건너뜀)
df_raw <- read_excel("Data/Training folder/CASE 006_IgD_L.xlsx",
                     sheet = 1,
                     skip  = 1,
                     col_names = TRUE)

# 2) 컬럼명 설정: No, 검사항목, 검체명, 참고치, 그리고 나머지는 날짜·시간
#    - 첫 5개 컬럼명 지정
colnames(df_raw)[1:5] <- c("No", "검사항목", "검체명", "참고치", "dummy")
df_raw
# 3) 불필요한 dummy 컬럼 제거
df_clean <- df_raw %>%
  select(-dummy) %>%
  mutate(across(No, as.integer))

# NA 컬럼명 처리: NA를 문자 "NA"로 변환하여 에러 방지
names(df_clean) <- make.names(names(df_clean), unique = TRUE)

# 4-b) 날짜·시간 컬럼명 복원: syntactic names을 "YYYY-MM-DD HH:MM" 형태로 변환
library(stringr)
orig_names <- names(df_clean)
new_names <- orig_names
# Apply only to measurement-time columns (5th onward)
for(i in 5:length(orig_names)) {
  nm <- orig_names[i]
  # Remove leading 'X' and split on one-or-more dots
  parts <- str_split(str_remove(nm, "^X"), "[.]+")[[1]]
  if(length(parts) >= 5) {
    new_names[i] <- sprintf("%s-%s-%s %s:%s",
                            parts[1], parts[2], parts[3],
                            parts[4], parts[5])
  }
}
names(df_clean) <- new_names

# 4-a) 측정값 컬럼을 모두 문자형으로 변환하여 pivot_longer 시 타입 충돌 방지
df_clean <- df_clean %>%
  mutate(across(5:ncol(.), as.character))

# 4) 긴 형태로 변환
df_tidy <- df_clean %>%
  pivot_longer(
    cols       = 5:ncol(df_clean),
    names_to   = "측정시각",
    values_to  = "값"
  )

# 5-a) pivot 후 값 컬럼을 숫자형으로 변환
df_tidy <- df_tidy %>%
  mutate(값 = as.numeric(값))

# 5) 문자열로 된 측정시각 파싱
df_tidy <- df_tidy %>%
  mutate(
    측정시각 = as.POSIXct(측정시각,
                   format = "%Y-%m-%d %H:%M",
                   tz = "Asia/Seoul"),
    날짜     = as.Date(측정시각),
    시간     = format(측정시각, "%H:%M")
  )

# 6) 2025년 3월 7일 이전 데이터만 추출
df_tidy <- df_tidy %>%
  filter(날짜 > as.Date("2024-11-05")) %>%
  filter(No < 4) %>%
  select(검체명, 측정시각, 값) %>%
    group_by(검체명) %>%
    mutate(n = row_number()) %>%
    ungroup() %>%
    mutate(Sequence = ifelse(n == 1, 'Baseline', paste0('followup_', n - 1))) %>%
    mutate(Result = case_when(
        검체명 == "Monoclonal peak" ~ "M_protein",
        검체명 == "Kappa light chain 정량" ~ "Kappa",
        검체명 == "Lambda light chain 정량" ~ "Lambda",
        NA ~ "Other"
    )) %>%
    select(Sequence, 값, Result, n) %>%
    rename(Value = 값)
 

# 결과 확인
head(df_tidy)

# Baseline 데이터만 추출하여 wide 형식으로 변환
baseline_df <- df_tidy %>%
  filter(Sequence == 'Baseline') %>%
  pivot_wider(names_from = Result, values_from = Value)
# 환자 유형 판정

baseline_df <- baseline_df %>%
  mutate(PatientType = case_when(
    M_protein >= 0.5 & Kappa > Lambda ~ 'IgG_kappa',
    M_protein >= 0.5 & Kappa <= Lambda ~ 'IgG_lambda',
    M_protein < 0.5 & (Kappa - Lambda) > 100 & Kappa > Lambda ~ 'LCD_kappa',
    M_protein < 0.5 & (Kappa - Lambda) > 100 & Kappa <= Lambda ~ 'LCD_lambda',
    TRUE ~ 'Unclassified'
  )) %>%
  select(PatientType) %>%
  pull()


# 7) 전체 데이터를 wide 형식으로 변환하여 M_protein 값만 추출
wide_df <- df_tidy %>%
  pivot_wider(names_from = Result, values_from = Value) %>%
  mutate(PatientType = baseline_df)

# 8) baseline M_protein 값 저장
base_m <- wide_df %>% filter(Sequence == "Baseline") %>% pull(M_protein)

# 9) IgG_kappa 환자만 필터링
igg_df <- wide_df

# 10) M_protein 변화율 계산 및 반응 정의
igg_df_tidy <- igg_df %>%
  arrange(n) %>%
  filter(!is.na(M_protein)) %>%
  mutate(
    # 10-a) 가장 작은 M_protein 값을 기준으로 시점별 추세 정의
    Trend = case_when(
      row_number() < which.min(M_protein)  ~ "decreasing",
      row_number() > which.min(M_protein)  ~ "increasing",
      TRUE                                 ~ "stable"
    ),
    pct_change = (base_m - M_protein) / base_m * 100,
        # 10-b) 지금까지의 M_protein 중 최소값 계산 (이전 행까지)
    prev_min = cummin(lag(M_protein, default = first(M_protein))),
    Response = case_when(
      M_protein == 0                            ~ "CR",
      !is.na(prev_min) & (M_protein - prev_min) > 0.45        ~ "Progression", # 반올림 해서 0.5 로 잡으신듯?
      Trend == "decreasing" & pct_change > 85                           ~ "VGPR",
      Trend == "decreasing" & pct_change > 45                           ~ "PR",
      Trend == "decreasing" & pct_change > 15                           ~ "MR",
      TRUE                                      ~ NA
    ),
    # 11) 두 번 연속 동일한 반응이 있을 때 최종 ConfirmedResponse로 지정
    ConfirmedResponse = ifelse(Response == lag(Response), Response, NA_character_)
  )

# 12) confirmedResponse 기준, VGPR은 즉시 반영, CR은 이전 최종 값 유지, progression 이후 중단
# 12) confirmedResponse가 있는 경우 바로 반영, VGPR은 즉시 반영, CR은 이전 최종 값을 유지, progression 이후 중단
final_resp <- character(nrow(igg_df_tidy))
prog_flag <- FALSE

for (i in seq_len(nrow(igg_df_tidy))) {
  if (prog_flag) {
    final_resp[i] <- NA_character_
  } else if (!is.na(igg_df_tidy$ConfirmedResponse[i])) {
    # ConfirmedResponse가 있으면 바로 사용, progression은 이전 시점에도 표시
    if (igg_df_tidy$ConfirmedResponse[i] == "Progression") {
      if (i > 1) {
        final_resp[i - 1] <- "Progression"
      }
      prog_flag <- TRUE
    }
    final_resp[i] <- igg_df_tidy$ConfirmedResponse[i]
  } else if (i > 1 && !is.na(final_resp[i - 1]) && final_resp[i - 1] == "CR") {
    # 이전에 CR이 관찰된 경우 progression 전까지 유지
    final_resp[i] <- "CR"
  } else if (!is.na(igg_df_tidy$Response[i]) && igg_df_tidy$Response[i] == "VGPR") {
    final_resp[i] <- "VGPR"
  } else if (!is.na(igg_df_tidy$Response[i]) && igg_df_tidy$Response[i] == "CR") {
    final_resp[i] <- if (i > 1) final_resp[i - 1] else NA_character_
  } else {
    # 그 외는 Response 그대로
    final_resp[i] <- igg_df_tidy$Response[i]
  }
}
igg_df_final <- igg_df_tidy %>%
  mutate(FinalResponse = final_resp)

# 13) 결과 출력
print(igg_df_final %>% select(Sequence, M_protein, pct_change, Trend, Response, ConfirmedResponse, FinalResponse), n = Inf)

