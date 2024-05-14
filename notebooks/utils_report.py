import pandas as pd


def construct_cta_report_test_set():

    from section_parser import section_text, section_text2

    df = pd.read_csv('../data/ct_report/LlamaPrompts1000.csv').rename(columns={'Reports': 'report'})

    def section_fn(s):
        sections, section_names, _ = section_text(s)
        sections = dict(zip(section_names, sections))
        return sections
    df['sections'] = df['report'].apply(section_fn)


    def section_fn(row):
        if 'findings' not in row:
            return {}
        
        sections = section_text2('\n' + row['findings'])
        sections = {k.lower(): v for k, v in sections.items() if k!='preamble'}
        return sections
    df['findings_sections'] = df['sections'].apply(section_fn)


    def construct_report_short(row):
        sections = row['sections']
        s = ''
        if 'findings' in sections:
            s += 'FINDINGS:\n\n'
            s += sections['findings']
            s += '\n\n'
        if 'impression' in sections:
            s += 'IMPRESSION:\n\n'
            s += sections['impression']
            s += '\n\n'
        s = s.strip()
        return s
    df['report_short'] = df.apply(construct_report_short, axis=1)
    print(len(df))

    ## save test set report
    df['id'] = df.index
    df[['id', 'report_short']].rename(columns={'report_short': 'report'}).to_json('/data/vision/polina/scratch/wpq/github/cxrpe/notebooks/prompts/classify_pe/test_set.json', orient='records', indent=4)


    ## save icl examples
    labels = [
        (0, "Pulmonary vasculature is well opacified to the\nsubsegmental level without filling defect to indicate a pulmonary embolus.", "negative"),
        (1, "Pulmonary vasculature is otherwise well opacified to the subsegmental level\nwithout filling defect to indicate a pulmonary embolus.", "negative"),
        (120, "Chronic appearing pulmonary emboli involving the distal right main\npulmonary artery, and multiple lobar, segmental and subsegmental branches of\nthe bilateral upper and lower lobes.", "negative"), # only account acute pe as positive
        (146, "1. No acute pulmonary embolism.\n2. Small residual nonocclusive filling defect representing chronic, organizing\nthrombus in a segmental artery feeding the right lower lobe.", "negative"), # since its chronic
        (147, "No\nfilling defect is seen within the pulmonary arterial tree to suggest the\npresence of a pulmonary embolism.", "negative"),
        (186, "Filling defect in single subsegmental branch of a left upper lobe pulmonary\nartery is might present in subsegmental pulmonary embolism, but may be\nsecondary to motion.", "negative"), # because this example not very sure, can be due to motion.
        (188, "Extensive acute bilateral pulmonary emboli extending from the right and left\ndistal main pulmonary arteries to the lobar, segmental, and subsegmental\nbranches of all lobes with evidence of right heart strain.", "positive"),
        (189, "Subsegmental nonocclusive pulmonary embolism in bilateral lung bases and\npossibly right upper lobe.", "positive"),
        (999, "There are extensive left upper, left lower, and right\nupper lobe pulmonary emboli. There are possibly tiny right lower lobe lateral\nand posterior basal segment emboli, though evaluation is limited by\nrespiratory motion artifact these levels.", "positive"), # second equivacol, but first is definitely positives
        (998, "The pulmonary arteries are well opacified to the subsegmental level, with no\nevidence of filling defect within the main, right, left, lobar or segmental\npulmonary arteries.", "negative"),
    ]

    icl_examples = []

    for idx, reference, label in labels:
        icl_examples.append({
            'idx': idx,
            'reference': reference,
            'label': label,
            'report': df.iloc[idx]['report_short'],
        })

    # pd.DataFrame(icl_examples).to_json('/data/vision/polina/scratch/wpq/github/cxrpe/notebooks/prompts/classify_pe/test_set.json', orient='records', indent=4)


