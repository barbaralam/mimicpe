## https://github.com/MIT-LCP/mimic-code 
#
# 


import re

blacklist_section_names = [
    'non-vascular',
    'abdomen',
    'abdomen / pelvis',
    'abdomen and pelvis',
    'abdomen pelvis',
    'abdomen/pelvis',
    'ap',
    'arteries',
    'amniotic fluid',
    'barium consistencies',
    'biopsy',
    'bmd',
    'bypass grafts',
    'carotid stenosis reference',
    'cervical spine',
    'chest',
    'dedicated hip',
    'dedicated left hip',
    'duodenum',
    'dob',
    'entire pelvis screening',
    'esophagus',
    'face',
    'flair',
    'general',
    'head',
    'head and neck',
    'iv contrast',
    'ivc',
    'ivcm',
    'ivcs',
    'kidneys',
    'left',
    'left  renal vein',
    'left anterior descending (lad)',
    'left circumflex (lcx)',
    'left fallopian tube',
    'left main',
    'left renal artery',
    'lower extremities',
    'lumbar spine',
    'mr orbits',
    'mra head',
    'mra neck',
    'mri brain',
    'mrv',
    'mrv head',
    'musculoskeletal',
    'musculoskeletal/lower extremities',
    'musculoskeletal/soft tissue',
    'musculoskeletal/soft tissues',
    'musculoskeletal/soft tissues/lower extremities',
    'neck',
    'nonvascular',
    'peripheral zone',
    'ri',
    'right',
    'right coronary artery (rca)',
    'right fallopian tube',
    'right foot',
    'right renal artery',
    'right renal vein',
    'soft tissues',
    'spect/ct',
    'spine',
    'stomach',
    'swi',
    'thoracic and lumbar spine',
    'thoracic spine',
    'thoracic spine / lumbar spine',
    'tips',
    'tmj',
    'transition zone',
    'ureters',
    'vascular',
    'veins',
    'whole lung',
    'adrenal glands',
    'angiogram',
    'biliary',
    'bones/soft tissues',
    'bones and soft tissues',
    'bowel',
    'kidneys/ureters',
    'left renal vein',
    'left ribs',
    'left shoulder',
    'liver',
    'limited abdomen',
    'lymph nodes',
    'maxillofacial',
    'mri sinus',
    'non-vascular',
    'pancreas',
    'pelvis',
    'peritoneum',
    'spleen',
    'ankle',
    'axillae',
    'bones',
    'bowel/peritoneum',
    'brain',
    'brain mri',
    'brain/extra-axial spaces',
    'cardiac',
    'cervical and thoracic spine',
    'ct head',
    'ct perfusion',
    'ctv head',
    'face (sinus)',
    'foot',
    'head ct',
    'kidneys/ureters/bladder',
    'left foot',
    'left lower extremity screening',
    'left wrist',
    'limitations',
    'lower thorax',
    'ls spine',
    'lungs/pleura',
    'lungs/airways',
    'lvad',
    'mediastinum',
    'mra brain',
    'orbits',
    'other',
    'paranasal sinuses/mastoids',
    'paraspinal soft tissues',
    'pre-watchman left atrial appendage measurements',
    'pulmonary angiogram',
    'pulmonary nodules',
    'pulmonary arteries',
    'placenta',
    'reproductive organs',
    'right elbow',
    'right wrist',
    'right ribs',
    'soft tissue',
    'soft tissue/axillae',
    'thoracolumbar spine',
    'total spine',
    'tumor',
    'upper abdomen',
    'vasculature',
    # below are cta reports
    'heart and vasculature',
    'axilla, hila, and mediastinum',
    'pleural spaces',
    'base of neck',
    'adrenals',
    'hepatobiliary',
    'gastrointestinal',
    'urinary',
    'osseous structures',
    'bone windows',
    'pleura',
    'dlp',
    'cta',
    'dlp',
    'lungs',
    'chest cage',
    'hila',
    'retroperitoneum',
    'lung',
    'neck, thoracic inlet, axillae, chest wall',
    'bone window',
    'airways',
    'study',
    'heart',
    'heart and vessels',
    'osseous/soft tissues',
    'vessels',
    'pulmonary arteries/aorta',
    'aorta',
    'neck, thoracic inlet, axillae',
    'pulmonary parenchyma',
    'chest wall and bones',
    'chest perimeter',
    'cardio-mediastinum',
    'thoracic lymph nodes',
    'lungs, airways, pleurae',
    'thyroid',
    'soft tissue of the chest cage',
    'mediastinum and hila',
    'chest wall',
    'thoracic aorta',
    'neck, thoracic inlet, axillae and chest wall',
    'thoracic inlet',
    'breast and axilla  ',
    'bones and chest wall ',
    'airways/lungs',
    'iliofemoral arteries',
    'subclavian arteries',
    'osseous structures/soft tissues',
    'bone',
    'spaces',
    'measurements',
    'heart/vasculature',
    'mediastinum/lymph nodes',
    'bones/chest wall',
    'venous',
    'mediastinum/hila',
    'base of neck/soft tissues of chest',
    ' maximal and minimal diameters from imaging lab',
    'maximal and minimal diameters from imaging lab',
    'd lp',
    'pulmonary arteries/aorta/heart',
    'lung parenchyma',
    'partially visualized abdomen',
    ' pleural spaces',
    'sof tissues',
    'mesentery and retroperitoneum',
    'thoracic soft tissues',
    'cardiac/pulmonary arteries/aorta',
    'base of neck and chest wall',
    ' neck, thoracic inlet, axillae and chest wall',
    'retroperitoneum/mesentery',
    'miscellaneous',
    'soft tissue/bones',
    'abdominal aorta',
    'iliacs and femorals',
    'axilla, hila',
    'note',
    '-parenchyma',
    '-airways',
    '-vessels',
    'lines and tubes',
    'lungs and airways',
    'pleura/pericardium',
    'esophagus and neck',
    'mediastinum/heart',
    ' heart and vasculature',
    ' axilla, hila, and mediastinum',
    'endoleak',
    'airway',
    'solid organs',
    'lower chest',
    'axilla, hila, mediastinum',
    'pleural',
]

blacklist_section_names = [
    ('\n'+x.upper()+':', '\n'+x.capitalize()+':') for x in blacklist_section_names
]


def combine_equivalent_sections(section_names, sections):
    """
        e.g., 
            section_names = ['findings', 'incidental findings']
            sections = ['1', '2']
            -> 
            section_naems = ['findings']
            sections = ['1\n2']
    """

    same_section_names = [
        ('findings', 'incidental findings', 'chest findings', 'chest ct findings', 'abdomen findings', 'neck findings', 'abdomen and pelvis findings', 'abdomen/pelvis findings', 'abdominal findings', 'vascular findings', 'non-vascular findings', 'ct abdomen/pelvis', 'chest and abdomen vascular', 'ct chest', 'ct chest findings', 'extracardiac findings', 'non vascular findings', 'nonvascular findings', 'non-coronary cardiac findings', 'head ct findings', 'cervical spine findings', 'cervical spine ct findings',
        # cta
        'cta',
        'cta thorax',
        'cta head',
        'cta neck',
        'cta chest',
        'ct thorax',
        'ct of the thorax',
        'ct abdomen',
        'ct pelvis',
        'ct pulmonary angiogram',
        'ct abdomen with contrast',
        'ct pelvis with contrast',
        'ct angiogram',
        'ct of the chest',
        'ct abdomen with intravenous contrast',
        'ct pelvis with intravenous contrast',
        'ct of the chest with intravenous contrast',
        'ct of the abdomen with intravenous contrast',
        'ct of the pelvis with intravenous contrast',
        'ct of the abdomen with',
        'ct of the abdomen with contrast',
        'ct pulmonary angiogram without and with iv contrast',
        'cta of the of the chest without and with intravenous contrast',
        'cta the abdomen',
        'cta mesentery',
        'cta chest with contrast and reconstructions',
        'cta abdomen',
        'cta pelvis',
        'cta of the chest without and with intravenous contrast',
        'cta chest with contrast',
        'cta chest', 
        'chest cta',
        'ct of the chest without and with intravenous contrast', 
        'ct chest without and with iv contrast', 
        'ct chest with and without contrast', 
        'ct abdomen with iv contrast', 
        'cta of the chest', 
        'ct chest with contrast', 
        'ct of the chest with and without intravenous contrast',
        ), # merge later sections into the first section listed.
    ]

    for same_sections in same_section_names:

        section_name_canonical = same_sections[0]

        sections_combine = list(filter(lambda l: l[0] in same_sections, zip(section_names, sections)))
        if not (len(sections_combine) > 1 or (len(sections_combine)==1 and sections_combine[0][0]!=section_name_canonical)):
            continue
        sections_rest = list(filter(lambda l: l[0] not in same_sections, zip(section_names, sections)))

        idx = None
        for i in range(len(section_names)):
            if section_names[i] in same_sections:
                idx = i
                break

        t = (section_name_canonical, '\n'.join([f'{k.upper()}:\n{v}' if k!=section_name_canonical else v for k, v in sections_combine]))
        sections_rest.insert(idx, t)
            
        
        section_names, sections = list(zip(*sections_rest))
        
    return section_names, sections



def section_text(text):
    """Splits text into sections.

    Assumes text is in a radiology report format, e.g.:

        COMPARISON:  Chest radiograph dated XYZ.

        IMPRESSION:  ABC...

    Given text like this, it will output text from each section, 
    where the section type is determined by the all caps header.

    Returns a three element tuple:
        sections - list containing the text of each section
        section_names - a normalized version of the section name
        section_idx - list of start indices of the text in the section
    """
    p_section = re.compile(
        r'\n([A-Z ()/,-]+):\s', re.DOTALL) # wpq: change from r'\n ([A-Z ()/,-]+):\s'!

    # wpq: substitute capitalized section names that should not be section names,
    # e.g., they are typically sub-sections in FINDINGS sections
    for s, t in blacklist_section_names:
        text = text.replace(s, t)

    sections = list()
    section_names = list()
    section_idx = list()

    idx = 0
    s = p_section.search(text, idx)

    if s:
        sections.append(text[0:s.start(1)])
        section_names.append('preamble')
        section_idx.append(0)

        while s:
            current_section = s.group(1).lower()
            # get the start of the text for this section
            idx_start = s.end()
            # skip past the first newline to avoid some bad parses
            idx_skip = text[idx_start:].find('\n')
            if idx_skip == -1:
                idx_skip = 0

            s = p_section.search(text, idx_start + idx_skip)

            if s is None:
                idx_end = len(text)
            else:
                idx_end = s.start()

            sections.append(text[idx_start:idx_end])
            section_names.append(current_section)
            section_idx.append(idx_start)

    else:
        sections.append(text)
        section_names.append('full report')
        section_idx.append(0)


    section_names = normalize_section_names_v2(section_names)
    section_names, sections = combine_equivalent_sections(section_names, sections)

    # section_names = normalize_section_names(section_names)

    # # remove empty sections
    # # this handles when the report starts with a finding-like statement
    # #  .. but this statement is not a section, more like a report title
    # #  e.g. p10/p10103318/s57408307
    # #    CHEST, PA LATERAL:
    # #
    # #    INDICATION:   This is the actual section ....
    # # it also helps when there are multiple findings sections
    # # usually one is empty
    # for i in reversed(range(len(section_names))):
    #     if section_names[i] in ('impression', 'findings'):
    #         if sections[i].strip() == '':
    #             sections.pop(i)
    #             section_names.pop(i)
    #             section_idx.pop(i)

    if ('impression' not in section_names) & ('findings' not in section_names):
        # create a new section for the final paragraph
        if '\n \n' in sections[-1]:
            sections.append('\n \n'.join(sections[-1].split('\n \n')[1:]))
            sections[-2] = sections[-2].split('\n \n')[0]
            section_names.append('last_paragraph')
            section_idx.append(section_idx[-1] + len(sections[-2]))

    return sections, section_names, section_idx



def section_text2(text):
    """now section names could be lower case as well. """
    
    p_section = re.compile(
        r'\n([a-zA-Z ()/,-,&]+):\s', re.DOTALL) # wpq: change from r'\n ([A-Z ()/,-]+):\s'!

    sections = list()
    section_names = list()
    section_idx = list()

    idx = 0
    s = p_section.search(text, idx)

    if s:
        sections.append(text[0:s.start(1)])
        section_names.append('preamble')
        section_idx.append(0)

        while s:
            current_section = s.group(1)
            # get the start of the text for this section
            idx_start = s.end()
            # skip past the first newline to avoid some bad parses
            idx_skip = text[idx_start:].find('\n')
            if idx_skip == -1:
                idx_skip = 0

            s = p_section.search(text, idx_start + idx_skip)

            if s is None:
                idx_end = len(text)
            else:
                idx_end = s.start()

            sections.append(text[idx_start:idx_end])
            section_names.append(current_section)
            section_idx.append(idx_start)

    else:
        sections.append(text)
        section_names.append('full report')
        section_idx.append(0)
        
    return dict(zip(section_names, sections))


frequent_sections = {
    "preamble": "preamble",  # 227885
    "impression": "impression",  # 187759
    "comparison": "comparison",  # 154647
    "indication": "indication",  # 153730
    "findings": "findings",  # 149842
    "examination": "examination",  # 94094
    "technique": "technique",  # 81402
    "history": "history",  # 45624
    "comparisons": "comparison",  # 8686
    "clinical history": "history",  # 7121
    "reason for examination": "indication",  # 5845
    "notification": "notification",  # 5749
    "reason for exam": "indication",  # 4430
    "clinical information": "history",  # 4024
    "exam": "examination",  # 3907
    "clinical indication": "indication",  # 1945
    "conclusion": "impression",  # 1802
    "chest, two views": "findings",  # 1735
    "recommendation(s)": "recommendations",  # 1700
    "type of examination": "examination",  # 1678
    "reference exam": "comparison",  # 347
    "patient history": "history",  # 251
    "addendum": "addendum",  # 183
    "comparison exam": "comparison",  # 163
    "date": "date",  # 108
    "comment": "comment",  # 88
    "findings and impression": "impression",  # 87
    "wet read": "wet read",  # 83
    "comparison film": "comparison",  # 79
    "recommendations": "recommendations",  # 72
    "findings/impression": "impression",  # 47
    "pfi": "history",
    'recommendation': 'recommendations',
    'wetread': 'wet read',
    'ndication': 'impression',  # 1
    'impressions': 'impression', # wpq added
    'impresson': 'impression',  # 2
    'imprression': 'impression',  # 1
    'imoression': 'impression',  # 1
    'impressoin': 'impression',  # 1
    'imprssion': 'impression',  # 1
    'impresion': 'impression',  # 1
    'imperssion': 'impression',  # 1
    'mpression': 'impression',  # 1
    'impession': 'impression',  # 3
    'findings/ impression': 'impression',  # ,1
    'finding': 'findings',  # ,8
    'findins': 'findings',
    'findindgs': 'findings',  # ,1
    'findgings': 'findings',  # ,1
    'findngs': 'findings',  # ,1
    'findnings': 'findings',  # ,1
    'finidngs': 'findings',  # ,2
    'idication': 'indication',  # ,1
    'reference findings': 'findings',  # ,1
    'comparision': 'comparison',  # ,2
    'comparsion': 'comparison',  # ,1
    'comparrison': 'comparison',  # ,1
    'comparisions': 'comparison'  # ,1
}


def normalize_section_names_v2(section_names):
    """Just normalize section names. """

    for i, s in enumerate(section_names):
        if s in frequent_sections:
            section_names[i] = frequent_sections[s]
            continue

    return section_names



def normalize_section_names(section_names):
    # first, lower case all
    section_names = [s.lower().strip() for s in section_names]

    p_findings = [
        'chest',
        'portable',
        'pa and lateral',
        'lateral and pa',
        'ap and lateral',
        'lateral and ap',
        'frontal and',
        'two views',
        'frontal view',
        'pa view',
        'ap view',
        'one view',
        'lateral view',
        'bone window',
        'frontal upright',
        'frontal semi-upright',
        'ribs',
        'pa and lat'
    ]
    p_findings = re.compile('({})'.format('|'.join(p_findings)))

    main_sections = [
        'impression', 'findings', 'history', 'comparison',
        'addendum'
    ]
    for i, s in enumerate(section_names):
        if s in frequent_sections:
            section_names[i] = frequent_sections[s]
            continue

        main_flag = False
        for m in main_sections:
            if m in s:
                section_names[i] = m
                main_flag = True
                break
        if main_flag:
            continue

        m = p_findings.search(s)
        if m is not None:
            section_names[i] = 'findings'

        # if it looks like it is describing the entire study
        # it's equivalent to findings
        # group similar phrasings for impression

    return section_names


def custom_mimic_cxr_rules():
    custom_section_names = {
        's50913680': 'recommendations',  # files/p11/p11851243/s50913680.txt
        's59363654': 'examination',  # files/p12/p12128253/s59363654.txt
        's59279892': 'technique',  # files/p13/p13150370/s59279892.txt
        's59768032': 'recommendations',  # files/p13/p13249077/s59768032.txt
        's57936451': 'indication',  # files/p14/p14325424/s57936451.txt
        's50058765': 'indication',  # files/p14/p14731346/s50058765.txt
        's53356173': 'examination',  # files/p15/p15898350/s53356173.txt
        's53202765': 'technique',  # files/p16/p16076182/s53202765.txt
        's50808053': 'technique',  # files/p16/p16631485/s50808053.txt
        's51966317': 'indication',  # files/p10/p10817099/s51966317.txt
        's50743547': 'examination',  # files/p11/p11388341/s50743547.txt
        's56451190': 'note',  # files/p11/p11842879/s56451190.txt
        's59067458': 'recommendations',  # files/p11/p11984647/s59067458.txt
        's59215320': 'examination',  # files/p12/p12408912/s59215320.txt
        's55124749': 'indication',  # files/p12/p12428492/s55124749.txt
        's54365831': 'indication',  # files/p13/p13876470/s54365831.txt
        's59087630': 'recommendations',  # files/p14/p14267880/s59087630.txt
        's58157373': 'recommendations',  # files/p15/p15032392/s58157373.txt
        's56482935': 'recommendations',  # files/p15/p15388421/s56482935.txt
        's58375018': 'recommendations',  # files/p15/p15505556/s58375018.txt
        's54654948': 'indication',  # files/p17/p17090359/s54654948.txt
        's55157853': 'examination',  # files/p18/p18975498/s55157853.txt
        's51491012': 'history',  # files/p19/p19314266/s51491012.txt

    }

    custom_indices = {
        's50525523': [201, 349],  # files/p10/p10602608/s50525523.txt
        's57564132': [233, 554],  # files/p10/p10637168/s57564132.txt
        's59982525': [313, 717],  # files/p11/p11989982/s59982525.txt
        's53488209': [149, 475],  # files/p12/p12458657/s53488209.txt
        's54875119': [234, 988],  # files/p13/p13687044/s54875119.txt
        's50196495': [59, 399],  # files/p13/p13894879/s50196495.txt
        's56579911': [59, 218],  # files/p15/p15394326/s56579911.txt
        's52648681': [292, 631],  # files/p15/p15666238/s52648681.txt
        's59889364': [172, 453],  # files/p15/p15835529/s59889364.txt
        's53514462': [73, 377],  # files/p16/p16297706/s53514462.txt
        's59505494': [59, 450],  # files/p16/p16730991/s59505494.txt
        's53182247': [59, 412],  # files/p16/p16770442/s53182247.txt
        's51410602': [47, 320],  # files/p17/p17069955/s51410602.txt
        's56412866': [522, 822],  # files/p17/p17612000/s56412866.txt
        's54986978': [59, 306],  # files/p17/p17912487/s54986978.txt
        's59003148': [262, 505],  # files/p17/p17916384/s59003148.txt
        's57150433': [61, 394],  # files/p18/p18335791/s57150433.txt
        's56760320': [219, 457],  # files/p18/p18418794/s56760320.txt
        's59562049': [158, 348],  # files/p18/p18502016/s59562049.txt
        's52674888': [145, 296],  # files/p19/p19381919/s52674888.txt
        's55258338': [192, 568],  # files/p13/p13719117/s55258338.txt
        's59330497': [140, 655],  # files/p15/p15479218/s59330497.txt
        's52119491': [179, 454],  # files/p17/p17959278/s52119491.txt
        # below have no findings at all in the entire report
        's58235663': [0, 0],  # files/p11/p11573679/s58235663.txt
        's50798377': [0, 0],  # files/p12/p12632853/s50798377.txt
        's54168089': [0, 0],  # files/p14/p14463099/s54168089.txt
        's53071062': [0, 0],  # files/p15/p15774521/s53071062.txt
        's56724958': [0, 0],  # files/p16/p16175671/s56724958.txt
        's54231141': [0, 0],  # files/p16/p16312859/s54231141.txt
        's53607029': [0, 0],  # files/p17/p17603668/s53607029.txt
        's52035334': [0, 0],  # files/p19/p19349312/s52035334.txt
    }

    return custom_section_names, custom_indices