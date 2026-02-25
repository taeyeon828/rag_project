-- 01_schema.sql
-- Food Manufacturing Ops DB (demo-friendly)

CREATE TABLE IF NOT EXISTS product_master (
  product_id      SERIAL PRIMARY KEY,
  product_name    TEXT NOT NULL,
  category        TEXT NOT NULL,     -- 예: 김치/만두/소스
  allergen_flags  TEXT,              -- 예: "새우,밀,대두" (없으면 NULL)
  standard_weight_g INT NOT NULL
);

CREATE TABLE IF NOT EXISTS batch_production (
  batch_id        TEXT PRIMARY KEY,  -- 예: KIMCHI-20260211-A-01
  product_id      INT NOT NULL REFERENCES product_master(product_id),
  line            TEXT NOT NULL,      -- 라인명: A/B
  work_date       DATE NOT NULL,
  start_time      TIMESTAMP NOT NULL,
  end_time        TIMESTAMP NOT NULL,
  qty_planned     INT NOT NULL,
  qty_produced    INT NOT NULL,
  qty_defect      INT NOT NULL,
  defect_reason   TEXT               -- 주요 불량 사유(요약)
);

CREATE TABLE IF NOT EXISTS ccp_temperature_log (
  log_id          BIGSERIAL PRIMARY KEY,
  batch_id        TEXT NOT NULL REFERENCES batch_production(batch_id),
  ccp_point       TEXT NOT NULL,      -- 예: "가열", "냉각", "급속냉동"
  target_min_c    NUMERIC(5,2) NOT NULL,
  target_max_c    NUMERIC(5,2) NOT NULL,
  measured_c      NUMERIC(5,2) NOT NULL,
  measured_time   TIMESTAMP NOT NULL,
  status          TEXT NOT NULL        -- PASS/FAIL
);

CREATE TABLE IF NOT EXISTS qc_inspection (
  qc_id           BIGSERIAL PRIMARY KEY,
  batch_id        TEXT NOT NULL REFERENCES batch_production(batch_id),
  inspection_type TEXT NOT NULL,      -- 미생물/중량/관능/금속검출 등
  result_value    TEXT NOT NULL,      -- 수치/판정값(문자열로 단순화)
  spec           TEXT NOT NULL,      -- 규격
  judgment       TEXT NOT NULL,      -- PASS/FAIL
  inspector      TEXT NOT NULL,
  inspected_time TIMESTAMP NOT NULL,
  note           TEXT
);

CREATE TABLE IF NOT EXISTS sanitation_check (
  check_id        BIGSERIAL PRIMARY KEY,
  work_date       DATE NOT NULL,
  area            TEXT NOT NULL,      -- 예: "포장실", "혼합실", "출입구"
  item            TEXT NOT NULL,      -- 예: "손세척", "바닥청결", "소독농도"
  score           INT NOT NULL,       -- 0~100
  status          TEXT NOT NULL,      -- OK/NG
  checker         TEXT NOT NULL,
  note            TEXT
);

CREATE TABLE IF NOT EXISTS raw_material_lot (
  material_lot_id TEXT PRIMARY KEY,  -- 예: SHRIMP-LOT-20260205
  material_name   TEXT NOT NULL,     -- 예: "새우", "밀가루"
  supplier        TEXT NOT NULL,
  received_date   DATE NOT NULL,
  expiry_date     DATE NOT NULL,
  allergen_flags  TEXT               -- 예: "새우", "밀"
);

-- 배치와 원재료 LOT 연결(추적성)
CREATE TABLE IF NOT EXISTS batch_material_usage (
  batch_id        TEXT NOT NULL REFERENCES batch_production(batch_id),
  material_lot_id TEXT NOT NULL REFERENCES raw_material_lot(material_lot_id),
  qty_used_kg     NUMERIC(10,2) NOT NULL,
  PRIMARY KEY (batch_id, material_lot_id)
);

-- 고객 클레임/반품(식품에서 자주 질문 나오는 지표)
CREATE TABLE IF NOT EXISTS customer_complaint (
  complaint_id    BIGSERIAL PRIMARY KEY,
  received_date   DATE NOT NULL,
  product_name    TEXT NOT NULL,
  issue_type      TEXT NOT NULL,      -- 이물/포장불량/이취/유통기한/중량 등
  severity        TEXT NOT NULL,      -- Low/Medium/High
  suspected_batch_id TEXT,            -- 모르면 NULL
  note            TEXT
);

-- 조회 성능을 위한 인덱스(데모에서도 체감 좋음)
CREATE INDEX IF NOT EXISTS idx_batch_work_date ON batch_production(work_date);
CREATE INDEX IF NOT EXISTS idx_ccp_batch_time ON ccp_temperature_log(batch_id, measured_time);
CREATE INDEX IF NOT EXISTS idx_qc_batch_time ON qc_inspection(batch_id, inspected_time);
CREATE INDEX IF NOT EXISTS idx_sanitation_date ON sanitation_check(work_date);
CREATE TABLE IF NOT EXISTS production_daily (
  work_date date NOT NULL,
  line text NOT NULL,
  product text NOT NULL,
  qty_produced int NOT NULL,
  qty_defect int NOT NULL,
  PRIMARY KEY (work_date, line, product)
);

CREATE TABLE IF NOT EXISTS equipment_maintenance (
  event_time timestamp NOT NULL,
  equipment_id text NOT NULL,
  event_type text NOT NULL, -- 점검/고장/정비
  note text,
  PRIMARY KEY (event_time, equipment_id)
);
