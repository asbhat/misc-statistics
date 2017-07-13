
from scipy.stats import norm
from scipy.stats import fisher_exact
from scipy.stats import chi2_contingency

SAMPLE_SIZE_THRESHOLD = 50

def samples_to_contingency_table(samples):
    """
    Turns the samples array into a contingency table

    Assumes samples come in the pattern: (sample_total1, subsample1, sample_total2, subsample2, ...)

    Reverses contingency_table_to_samples()
    """
    return [[samples[i] - samples[i+1], samples[i+1]] for i in xrange(0, len(samples), 2)]

def contingency_table_to_samples(table):
    """
    Turns an M x 2 contingency_table into a samples array

    Reverses samples_to_contingency_table()
    """
    return [r[0] + r[1] if i == 0 else r[1] for r in table for i in xrange(len(r))]

def two_proportion_z_score(control_size, control_conversion, attribute_size, attribute_conversion, pooled_sample=True):
    """
    Calculate the z-score between two proportions (often used for A/B testing)

    Answers the question: Is the difference in conversion rate between two sample groups due to chance? Or is it a significant difference?

    Parameters
    -----------
    attribute vs. control : The labels are arbitrary and you will get the same result with opposite sign if the values are swapped

    attribute/control_size : total number of people in this sample
    attribute/control_conversion : subgroup of people in the sample (should always be <= attribute/control_size)

    pooled_sample : Choose between pooling the sample for calculating the Standard Error, or not
        This assumes the null hypothesis (attribute_pct == control_pct) and can be derived from the non-pooled-sample Standard Error
        Most resources I found had the pooled sample as standard (so I defaulted to it)

    Example
    -----------
    66,471 people were part of a study. Of those 5,201 were shown an ad.
    And of the 5,201 67 bought the product.
    Of the unexposed people (66,471 - 5,201 = 61,270) 200 bought the product.
    Are the exposed and unexposed conversion rates significantly different?

    control_size = 61,270
    control_conversion = 200
    attribute_size = 5,201
    attribute_conversion = 67

    zscore = 10.52848...

    Assumptions
    -----------
    - The test statistic follows a Normal Distribution (this is generally true the larger the sample per the Central Limit Theorem)
        - This is an approximate measure (unlike "exact" tests) approximations are used for mean and variance measures
        - For small sample sizes use Fisher's Exact Test
    - Each sample is independent
    - The sample is obtained by simple random sampling (every member of the total population has an equal chance of being selected)


    For more information, see:
    - https://onlinecourses.science.psu.edu/stat414/node/268
    - http://stattrek.com/hypothesis-test/difference-in-proportions.aspx?Tutorial=AP
    """
    attribute_pct = float(attribute_conversion) / attribute_size
    control_pct = float(control_conversion) / control_size

    if pooled_sample:
        total_pct = float(attribute_conversion + control_conversion) / (attribute_size + control_size)
        standard_error = ( total_pct * (1. - total_pct) * (1. / attribute_size + 1. / control_size) ) ** .5
    else:
        standard_error = ( (attribute_pct * (1. - attribute_pct) / attribute_size) + (control_pct * (1. - control_pct) / control_size) ) ** .5

    zscore = (attribute_pct - control_pct) / standard_error
    return zscore

def fishers_exact_test(control_size, control_conversion, attribute_size, attribute_conversion, alternative='two-sided'):
    """
    Fisher's Exact Test

    Mostly used as an alternative to the chi-squared test with low sample sizes

    Since this test is based on a hypergeometric distribution (not symmetrical) two-tailed p-values != 2 * one-tailed
        the p-value for each table with a lower probablity must be summed
        this is a pain, so I've just wrapped scipy's implementation

    Parameters
    ----------
    alternative : ['two-sided', 'less', 'greater']
        represents the alternative hypothesis (opposite to the null hypothesis)


    For more information, see:
    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html
    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html
    - https://en.wikipedia.org/wiki/Fisher%27s_exact_test
    """
    samples = [control_size, control_conversion, attribute_size, attribute_conversion]
    contingency_table = samples_to_contingency_table(samples)
    oddsratio, pvalue = fisher_exact(contingency_table, alternative)
    return pvalue

def p_value_from_z_score(zscore, two_tailed=True):
    pvalue = norm.sf( abs(zscore) )
    if two_tailed:
        pvalue *= 2.
    return pvalue

def two_proportion_p_value(control_size, control_conversion, attribute_size, attribute_conversion, pooled_sample=True, two_tailed=True):
    samples = [control_size, control_conversion, attribute_size, attribute_conversion]

    if any(map(lambda x: x < SAMPLE_SIZE_THRESHOLD, samples)):
        pvalue = fishers_exact_test(*samples)
    else:
        zscore = two_proportion_z_score(*samples, pooled_sample=pooled_sample)
        pvalue = p_value_from_z_score(zscore, two_tailed)

    return pvalue

def chi_squared_independence_test(arr, convert_to_table=False, yates_correction=False):
    """
    Chi Squared Test of Independence

    Answers the question: Did these data come from different distributions? Or are the differences due to chance?

    You'll notice on a 2x2 contingency table, the p-value is the same as the two_proportion_p_value()
    For consistency I'd recommend using the same SAMPLE_SIZE_THRESHOLD variable to determine if you have a "small" sample

    Parameters
    ----------------
    arr : Either pass the contingency table as an array of arrays, or pass a samples array
        i.e., samples = [control_size, control_conversion, attribute_size, attribute_conversion]
        e.g., samples = [374, 26, 210, 8] == contingency_table = [[348, 26], [202, 8]]

    convert_to_table : If you passed a samples array, you'll need this to be True, otherwise
        if you passed a contingency table directly, leave it as False

    Assumptions
    ----------------
    - The test statistic follows a Chi-Squared Distribution (this is generally true the larger the sample per the Central Limit Theorem)
        - This is an approximate measure (unlike "exact" tests) approximations are used for mean and variance measures
        - For small sample sizes use the use Fisher's Exact Test, or the Yates correction (see the "Yates Correction" section for more info)
    - Each sample is independent
    - The sample is obtained by simple random sampling (every member of the total population has an equal chance of being selected)

    Yates Correction
    ----------------
    This is a correction for small sample sizes in a 2x2 contingency table
    I would not recommend using this correction. It tends to be overly conservative (higher p-value than should be)
    Instead, for small samples I'd recommend using Fisher's Exact Test
    To read more about this see: http://www.statisticshowto.com/what-is-the-yates-correction/
        and see the sources near the end


    For more information, see:
    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html
    """
    if convert_to_table:
        table = samples_to_contingency_table(arr)
    else:
        table = arr

    chi2, pvalue, dof, expected = chi2_contingency(table, correction=yates_correction)
    return pvalue
