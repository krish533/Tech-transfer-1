from run_federal_rd_exposure_analysis import TOTAL_URL, FED_URL, OUTDIR, read_rank_xlsx, norm_name


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    total = read_rank_xlsx(TOTAL_URL, "total")
    fed = read_rank_xlsx(FED_URL, "fed")
    n = total.merge(fed, on="ncses_name", how="inner", validate="one_to_one")
    n["norm"] = n["ncses_name"].map(norm_name)
    n.to_csv(OUTDIR / "ncses_source_universe.csv", index=False)
    print("NCSES source-universe rows:", len(n))


if __name__ == "__main__":
    main()
