-- 02_seed.sql
-- Seed data for demo

INSERT INTO product_master (product_name, category, allergen_flags, standard_weight_g) VALUES
('배추김치', '김치', NULL, 500),
('고기만두', '만두', '밀,대두', 600),
('새우만두', '만두', '새우,밀,대두', 600)
ON CONFLICT DO NOTHING;

-- 제품 id 확보를 위해 간단 조회 기반으로 넣는 방식(고정값 가정 X)
-- 배치 데이터
INSERT INTO batch_production (batch_id, product_id, line, work_date, start_time, end_time, qty_planned, qty_produced, qty_defect, defect_reason)
SELECT
  'KIMCHI-20260209-A-01', pm.product_id, 'A', '2026-02-09',
  '2026-02-09 08:20:00', '2026-02-09 11:10:00',
  1200, 1200, 18, '중량 편차'
FROM product_master pm WHERE pm.product_name='배추김치'
ON CONFLICT DO NOTHING;

INSERT INTO batch_production (batch_id, product_id, line, work_date, start_time, end_time, qty_planned, qty_produced, qty_defect, defect_reason)
SELECT
  'MEATDUM-20260209-B-01', pm.product_id, 'B', '2026-02-09',
  '2026-02-09 09:00:00', '2026-02-09 12:05:00',
  900, 900, 7, '포장 실링 불량'
FROM product_master pm WHERE pm.product_name='고기만두'
ON CONFLICT DO NOTHING;

INSERT INTO batch_production (batch_id, product_id, line, work_date, start_time, end_time, qty_planned, qty_produced, qty_defect, defect_reason)
SELECT
  'KIMCHI-20260210-A-01', pm.product_id, 'A', '2026-02-10',
  '2026-02-10 08:15:00', '2026-02-10 11:00:00',
  1300, 1300, 12, '라벨 부착 위치 불량'
FROM product_master pm WHERE pm.product_name='배추김치'
ON CONFLICT DO NOTHING;

INSERT INTO batch_production (batch_id, product_id, line, work_date, start_time, end_time, qty_planned, qty_produced, qty_defect, defect_reason)
SELECT
  'MEATDUM-20260210-B-01', pm.product_id, 'B', '2026-02-10',
  '2026-02-10 09:05:00', '2026-02-10 12:20:00',
  950, 950, 11, '중량 편차'
FROM product_master pm WHERE pm.product_name='고기만두'
ON CONFLICT DO NOTHING;

INSERT INTO batch_production (batch_id, product_id, line, work_date, start_time, end_time, qty_planned, qty_produced, qty_defect, defect_reason)
SELECT
  'KIMCHI-20260211-A-01', pm.product_id, 'A', '2026-02-11',
  '2026-02-11 08:10:00', '2026-02-11 11:05:00',
  1250, 1250, 25, '실링 불량 증가'
FROM product_master pm WHERE pm.product_name='배추김치'
ON CONFLICT DO NOTHING;

-- CCP 온도 로그: 만두는 가열/냉각 같은 CCP가 자연스러움, 김치는 냉각/보관 온도 등을 데모로
INSERT INTO ccp_temperature_log (batch_id, ccp_point, target_min_c, target_max_c, measured_c, measured_time, status) VALUES
('MEATDUM-20260209-B-01','가열', 72.0, 78.0, 74.5, '2026-02-09 10:10:00','PASS'),
('MEATDUM-20260209-B-01','냉각',  0.0,  5.0,  6.2, '2026-02-09 11:30:00','FAIL'),
('MEATDUM-20260210-B-01','가열', 72.0, 78.0, 73.8, '2026-02-10 10:20:00','PASS'),
('MEATDUM-20260210-B-01','냉각',  0.0,  5.0,  4.6, '2026-02-10 11:50:00','PASS'),
('KIMCHI-20260211-A-01','저장',  0.0,  5.0,  5.8, '2026-02-11 10:40:00','FAIL');

-- QC 검사
INSERT INTO qc_inspection (batch_id, inspection_type, result_value, spec, judgment, inspector, inspected_time, note) VALUES
('KIMCHI-20260209-A-01','중량검사','496~505g','500g±10g','PASS','QC_김','2026-02-09 11:20:00',NULL),
('KIMCHI-20260210-A-01','관능검사','이취 없음','이취/이물 없음','PASS','QC_박','2026-02-10 11:15:00',NULL),
('KIMCHI-20260211-A-01','포장검사','실링 불량 2.0%','실링 불량 ≤ 0.5%','FAIL','QC_김','2026-02-11 11:10:00','실링 온도/압력 점검 필요'),
('MEATDUM-20260209-B-01','금속검출','정상','Fe/SUS 미검출','PASS','QC_최','2026-02-09 12:15:00',NULL),
('MEATDUM-20260209-B-01','미생물','적합','기준 적합','PASS','QC_최','2026-02-09 12:18:00',NULL);

-- 위생 점검(SSOP 느낌)
INSERT INTO sanitation_check (work_date, area, item, score, status, checker, note) VALUES
('2026-02-09','포장실','손세척/장갑 착용', 92, 'OK', '관리_이', NULL),
('2026-02-09','혼합실','바닥 청결', 85, 'OK', '관리_이', NULL),
('2026-02-10','포장실','소독 농도', 78, 'NG', '관리_정', '소독액 농도 재조정 필요'),
('2026-02-11','포장실','실링부 주변 청결', 74, 'NG', '관리_정', '실링부 분진/이물 가능성'),
('2026-02-11','출입구','에어샤워/점착매트', 90, 'OK', '관리_정', NULL);

-- 원재료 LOT
INSERT INTO raw_material_lot (material_lot_id, material_name, supplier, received_date, expiry_date, allergen_flags) VALUES
('CABBAGE-LOT-20260205','배추','농가A','2026-02-05','2026-02-20',NULL),
('FLOUR-LOT-20260201','밀가루','제분사B','2026-02-01','2026-06-01','밀'),
('SHRIMP-LOT-20260205','새우','수산C','2026-02-05','2026-03-05','새우'),
('SOY-LOT-20260128','대두','식자재D','2026-01-28','2026-05-28','대두')
ON CONFLICT DO NOTHING;

-- 배치-원재료 사용(추적성)
INSERT INTO batch_material_usage (batch_id, material_lot_id, qty_used_kg) VALUES
('KIMCHI-20260209-A-01','CABBAGE-LOT-20260205', 420.0),
('KIMCHI-20260210-A-01','CABBAGE-LOT-20260205', 450.0),
('KIMCHI-20260211-A-01','CABBAGE-LOT-20260205', 430.0),
('MEATDUM-20260209-B-01','FLOUR-LOT-20260201',   80.0),
('MEATDUM-20260210-B-01','FLOUR-LOT-20260201',   85.0),
('MEATDUM-20260210-B-01','SOY-LOT-20260128',     12.0)
ON CONFLICT DO NOTHING;

-- 고객 클레임
INSERT INTO customer_complaint (received_date, product_name, issue_type, severity, suspected_batch_id, note) VALUES
('2026-02-10','고기만두','포장불량','Medium','MEATDUM-20260209-B-01','실링이 약해 누수 발생'),
('2026-02-11','배추김치','이취','Low',NULL,'산미 강함(개인차 가능)'),
('2026-02-11','배추김치','포장불량','High','KIMCHI-20260211-A-01','실링 불량으로 누수/오염 가능')
;
INSERT INTO production_daily (work_date, line, product, qty_produced, qty_defect) VALUES
('2026-02-09','A','김치',1200,18),
('2026-02-09','B','만두',900,7),
('2026-02-10','A','김치',1300,12),
('2026-02-10','B','만두',950,11),
('2026-02-11','A','김치',1250,25)
ON CONFLICT DO NOTHING;

INSERT INTO equipment_maintenance (event_time, equipment_id, event_type, note) VALUES
('2026-02-10 09:10:00','MIXER-01','점검','베어링 소음 경미'),
('2026-02-10 14:25:00','PACK-02','고장','센서 오류로 라인 정지 8분'),
('2026-02-11 10:05:00','PACK-02','정비','센서 교체 완료')
ON CONFLICT DO NOTHING;
