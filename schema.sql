-- 코드 그룹
CREATE TABLE code_group (
  grp_code varchar(20) PRIMARY KEY,
  use_yn boolean DEFAULT true,
  disp_ord integer,
  del_yn boolean DEFAULT false,
  creator varchar(20),
  date_created timestamp DEFAULT now(),
  updater varchar(20),
  date_modified timestamp DEFAULT now()
);

COMMENT ON TABLE code_group IS '코드 그룹 관리';


-- 코드 마스터
CREATE TABLE code_master (
  code varchar(20) PRIMARY KEY,
  grp_code varchar(20) NOT NULL REFERENCES code_group(grp_code) ON DELETE CASCADE,
  add_info01 varchar(100),
  add_info02 varchar(100),
  add_info03 varchar(100),
  add_info04 varchar(100),
  use_yn boolean DEFAULT true,
  del_yn boolean DEFAULT false,
  disp_ord integer,
  creator varchar(20),
  date_created timestamp DEFAULT now(),
  updater varchar(20),
  date_modified timestamp DEFAULT now()
);

COMMENT ON TABLE code_master IS '코드정보 테이블';


-- 다국어 코드
CREATE TABLE code_lang (
  code_lng_id varchar(20) NOT NULL,
  type varchar(10) NOT NULL,
  lang varchar(2) NOT NULL,
  value varchar(100) NOT NULL,
  PRIMARY KEY (code_lng_id, type, lang)
);

COMMENT ON TABLE code_lang IS 'code_master 및 code_group 다국어 텍스트 정보';


-- 사용자 기본 정보
CREATE TABLE member_master (
  mem_id varchar(20) PRIMARY KEY,
  email varchar(100) NOT NULL,
  emp_no varchar(30),
  tel_no varchar(20),
  mobile_no varchar(20),
  enter_date varchar(8),
  leave_date varchar(8),
  img_url varchar(2000),
  fax_no varchar(20),
  gender char(1),
  birth_ymd varchar(8),
  lunar_yn boolean DEFAULT false,
  incharge_task varchar(500),
  alarm_yn boolean DEFAULT false,
  status_code varchar(20) NOT NULL,
  del_yn boolean DEFAULT false,
  creator varchar(20),
  date_created timestamp DEFAULT now(),
  updater varchar(20),
  date_modified timestamp DEFAULT now()
);

COMMENT ON TABLE member_master IS '사용자 기본 정보 테이블';


-- 사용자 다국어
CREATE TABLE member_lang (
  mem_id varchar(20) NOT NULL REFERENCES member_master(mem_id) ON DELETE CASCADE,
  type varchar(10) NOT NULL,
  lang varchar(2) NOT NULL,
  value varchar(300) NOT NULL,
  PRIMARY KEY (mem_id, type, lang)
);

COMMENT ON TABLE member_lang IS '사용자 다국어 텍스트 관리 테이블';


-- 조직 마스터
CREATE TABLE org_master (
  org_code varchar(20) PRIMARY KEY,
  up_org_code varchar(20) REFERENCES org_master(org_code) ON DELETE SET NULL,
  rep_org_code varchar(20),
  org_type char(1) NOT NULL,
  disp_ord integer,
  disp_yn boolean DEFAULT true,
  use_yn boolean DEFAULT true,
  del_yn boolean DEFAULT false,
  creator varchar(20),
  date_created timestamp DEFAULT now(),
  updater varchar(20),
  date_modified timestamp DEFAULT now()
);

COMMENT ON TABLE org_master IS '조직 관계 및 정보 테이블';


-- 조직 다국어
CREATE TABLE org_lang (
  org_code varchar(20) NOT NULL REFERENCES org_master(org_code) ON DELETE CASCADE,
  type varchar(3) NOT NULL,
  lang varchar(2) NOT NULL,
  value varchar(100) NOT NULL,
  PRIMARY KEY (org_code, type, lang)
);

COMMENT ON TABLE org_lang IS '조직 언어별 텍스트 관리 테이블';


-- 조직-사용자 연결
CREATE TABLE org_member_rel (
  org_code varchar(20) NOT NULL REFERENCES org_master(org_code) ON DELETE CASCADE,
  mem_id varchar(20) NOT NULL REFERENCES member_master(mem_id) ON DELETE CASCADE,
  jc_code varchar(20),
  jw_code varchar(20),
  jk_code varchar(20),
  jh_code varchar(20),
  jm_code varchar(20),
  jx_code varchar(20),
  rep_org_code varchar(20),
  priority smallint NOT NULL,
  disp_ord smallint,
  use_yn boolean DEFAULT true,
  creator varchar(20),
  date_created timestamp DEFAULT now(),
  updater varchar(20),
  date_modified timestamp DEFAULT now(),
  PRIMARY KEY (org_code, mem_id)
);

COMMENT ON TABLE org_member_rel IS '조직과 사용자 관계 관리';


-- 결재문서 카테고리
CREATE TABLE appr_docs_catg (
  adocs_catg_code varchar(20) PRIMARY KEY,
  up_adocs_catg_code varchar(20) REFERENCES appr_docs_catg(adocs_catg_code) ON DELETE SET NULL,
  disp_ord smallint,
  use_yn boolean DEFAULT true,
  del_yn boolean DEFAULT false
);

COMMENT ON TABLE appr_docs_catg IS '결재문서 양식 카테고리 관리';


-- 결재문서 마스터
CREATE TABLE appr_docs_master (
  adocs_code varchar(20) PRIMARY KEY,
  docs_type char(1),
  adocs_catg_code varchar(20) REFERENCES appr_docs_catg(adocs_catg_code),
  editor_yn boolean DEFAULT false,
  editor_body text,
  disp_yn boolean DEFAULT true,
  src_type varchar(10),
  opt_offc_yn boolean DEFAULT false,
  opt_vwr_yn boolean DEFAULT false,
  opt_rfr_yn boolean DEFAULT false,
  opt_alrm_rqs_type varchar(10),
  opt_mod_yn boolean DEFAULT false,
  opt_drft_cncl_yn boolean DEFAULT false,
  opt_appr_cncl_yn boolean DEFAULT false,
  opt_rwr_yn boolean DEFAULT false,
  opt_line_mod_yn boolean DEFAULT false,
  opt_no_type varchar(10),
  opt_recv_mod_yn boolean DEFAULT false,
  proc_all_yn boolean DEFAULT false,
  proc_adv_yn boolean DEFAULT false,
  proc_orjc_yn boolean DEFAULT false,
  proc_aggr_sqn_yn boolean DEFAULT false,
  proc_aggr_prl_yn boolean DEFAULT false,
  proc_aggr_sqn_cnt smallint,
  proc_aggr_prl_cnt smallint,
  fvrt_yn boolean DEFAULT false,
  line_type varchar(10),
  disp_ord smallint,
  use_yn boolean DEFAULT true,
  del_yn boolean DEFAULT false,
  creator varchar(20),
  date_created timestamp DEFAULT now(),
  updater varchar(20),
  date_modified timestamp DEFAULT now()
);

COMMENT ON TABLE appr_docs_master IS '결재 문서 양식 기본정보';


-- 결재문서 다국어
CREATE TABLE appr_docs_lang (
  adocs_lng_id varchar(20) NOT NULL,
  type varchar(3) NOT NULL,
  lang varchar(2) NOT NULL,
  value varchar(300),
  PRIMARY KEY (adocs_lng_id, type, lang)
);

COMMENT ON TABLE appr_docs_lang IS '결재 문서/카테고리 언어별 텍스트';


-- 결재 양식 권한 연결
CREATE TABLE appr_docs_target_rel (
  adocs_code varchar(20) NOT NULL REFERENCES appr_docs_master(adocs_code) ON DELETE CASCADE,
  rel_id varchar(20),
  rel_type varchar(10),
  PRIMARY KEY (adocs_code, rel_id)
);


-- 결재 양식 조회 권한 연결
CREATE TABLE appr_docs_select_rel (
  adocs_code varchar(20) NOT NULL REFERENCES appr_docs_master(adocs_code) ON DELETE CASCADE,
  rel_id varchar(20),
  rel_type varchar(10),
  PRIMARY KEY (adocs_code, rel_id)
);


-- 결재 마스터
CREATE TABLE appr_master (
  appr_id varchar(20) PRIMARY KEY,
  up_appr_id varchar(20),
  adocs_code varchar(20) NOT NULL REFERENCES appr_docs_master(adocs_code),
  appr_no varchar(100),
  title varchar(200),
  status_code varchar(10) NOT NULL,
  docs_type char(1),
  editor_yn boolean DEFAULT false,
  editor_body text,
  src_type varchar(10),
  drft_date timestamp,
  fnl_date timestamp,
  drft_mem_id varchar(20) REFERENCES member_master(mem_id),
  drft_org_code varchar(20) REFERENCES org_master(org_code),
  drft_rep_org_code varchar(20),
  drft_jx_code varchar(10),
  fnl_mem_id varchar(20),
  fnl_org_code varchar(20),
  fnl_rep_org_code varchar(20),
  fnl_jx_code varchar(10),
  del_yn boolean DEFAULT false,
  creator varchar(20),
  date_created timestamp DEFAULT now(),
  updater varchar(20),
  date_modified timestamp DEFAULT now()
);

COMMENT ON TABLE appr_master IS '결재 정보 마스터 테이블';


-- 결재 프로세스
CREATE TABLE appr_proc (
  aproc_id varchar(20) PRIMARY KEY,
  appr_id varchar(20) NOT NULL REFERENCES appr_master(appr_id) ON DELETE CASCADE,
  mem_id varchar(20) REFERENCES member_master(mem_id),
  org_code varchar(20) REFERENCES org_master(org_code),
  rep_org_code varchar(20),
  jx_code varchar(10),
  rl_mem_id varchar(20),
  rl_org_code varchar(20),
  rl_rep_org_code varchar(20),
  rl_jx_code varchar(10),
  inst_yn boolean DEFAULT false,
  proc_order smallint,
  proc_div_code varchar(10),
  proc_type_code varchar(10),
  proc_status_code varchar(10),
  proc_action_code varchar(10),
  proc_action_date timestamp,
  proc_recv_date timestamp,
  proc_check_date timestamp,
  proc_line_type varchar(10),
  mod_yn boolean DEFAULT true,
  creator varchar(20),
  date_created timestamp DEFAULT now(),
  updater varchar(20),
  date_modified timestamp DEFAULT now()
);

COMMENT ON TABLE appr_proc IS '결재 라인 정보 테이블';


-- 결재 다국어
CREATE TABLE appr_master_lang (
  appr_lng_id varchar(20) NOT NULL,
  type varchar(3) NOT NULL,
  lang varchar(2) NOT NULL,
  value varchar(300) NOT NULL,
  PRIMARY KEY (appr_lng_id, type, lang)
);

COMMENT ON TABLE appr_master_lang IS '결재 정보 언어별 텍스트 연결 테이블';
