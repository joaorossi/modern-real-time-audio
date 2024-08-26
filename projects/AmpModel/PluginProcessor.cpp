#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <cmath>

static const std::vector<mrta::ParameterInfo> ParameterInfos
{
    { Param::ID::Volume,  Param::Name::Volume,  "", 0.0f, 0.0, 1.f, 0.1f, 1.0f },
    { Param::ID::Tone,  Param::Name::Tone,  "", 0.0f, 0.0f, 1.f, 0.1f, 1.0f }
};

AmpModelProcessor::AmpModelProcessor() :
    parameterManager(*this, ProjectInfo::projectName, ParameterInfos)
{
    parameterManager.registerParameterCallback(Param::ID::Volume,
    [this] (float value, bool forced)
    {
        DBG(Param::Name::Volume + ": " + juce::String { value });
        // multiply by 0.8f because that's how it was trained
        if (forced)
            volume.setCurrentAndTargetValue(value * 0.8f);
        else
            volume.setTargetValue(value * 0.8f);
    });
    parameterManager.registerParameterCallback(Param::ID::Tone,
    [this] (float value, bool forced)
    {
        DBG(Param::Name::Tone + ": " + juce::String { value });
        // multiply by 0.8f because that's how it was trained
        if (forced)
            tone.setCurrentAndTargetValue(value * 0.8f);
        else
            tone.setTargetValue(value * 0.8f);
    });

    gru[0].load_parameters(gruParameters.params);
    gru[1].load_parameters(gruParameters.params);
}

AmpModelProcessor::~AmpModelProcessor()
{
}

void AmpModelProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    juce::uint32 numChannels { static_cast<juce::uint32>(std::max(getMainBusNumInputChannels(), getMainBusNumOutputChannels())) };
    volume.reset(sampleRate, 0.01f);
    tone.reset(sampleRate, 0.01f);
    parameterManager.updateParameters(true);
    nnInputBuffer.setSize(samplesPerBlock, INPUT_SIZE);
    nnOutputBuffer.setSize(samplesPerBlock, OUTPUT_SIZE);
    gru[0].reset_state();
    gru[1].reset_state();
}

void AmpModelProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& /*midiMessages*/)
{
    juce::ScopedNoDenormals noDenormals;
    parameterManager.updateParameters();

    const float * const * nn_input_read_ptr = nnInputBuffer.getArrayOfReadPointers();
    const float * const * nn_output_read_ptr = nnOutputBuffer.getArrayOfReadPointers();
    float * const * nn_input_write_ptr = nnInputBuffer.getArrayOfWritePointers();
    float * const * nn_output_write_ptr = nnOutputBuffer.getArrayOfWritePointers();
    const float * const * audio_read_ptr = buffer.getArrayOfReadPointers();
    float * const * audio_write_ptr = buffer.getArrayOfWritePointers();

    // first write input controls to nn_input_buffer
    for (size_t i = 0; i < buffer.getNumSamples(); ++i)
    {
        nn_input_write_ptr[i][1] = volume.getNextValue();
        nn_input_write_ptr[i][2] = tone.getNextValue();
    }

    for (size_t ch = 0; ch < std::min(buffer.getNumChannels(), 2); ++ch)
    {
        // initialise gru input with audio
        for (size_t i = 0; i < buffer.getNumSamples(); ++i)
        {
            nn_input_write_ptr[i][0] = audio_read_ptr[ch][i];
        }
        // process gru
        gru[ch].process(nn_output_write_ptr, nn_input_read_ptr, buffer.getNumSamples());
        // copy gru output to audio buffer
        for (size_t i = 0; i < buffer.getNumSamples(); ++i)
        {
            audio_write_ptr[ch][i] = nn_output_read_ptr[i][0];
        }
    }
}

void AmpModelProcessor::releaseResources()
{
}

void AmpModelProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    parameterManager.getStateInformation(destData);
}

void AmpModelProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    parameterManager.setStateInformation(data, sizeInBytes);
}

juce::AudioProcessorEditor* AmpModelProcessor::createEditor()
{
    return new AmpModelProcessorEditor(*this);
}

//==============================================================================
const juce::String AmpModelProcessor::getName() const { return JucePlugin_Name; }
bool AmpModelProcessor::acceptsMidi() const { return false; }
bool AmpModelProcessor::producesMidi() const { return false; }
bool AmpModelProcessor::isMidiEffect() const { return false; }
double AmpModelProcessor::getTailLengthSeconds() const { return 0.0; }
int AmpModelProcessor::getNumPrograms() { return 1; }
int AmpModelProcessor::getCurrentProgram() { return 0; }
void AmpModelProcessor::setCurrentProgram (int) { }
const juce::String AmpModelProcessor::getProgramName(int) { return {}; }
void AmpModelProcessor::changeProgramName(int, const juce::String&) { }
bool AmpModelProcessor::hasEditor() const { return true; }
//==============================================================================

//==============================================================================
// This creates new instances of the plugin..
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new AmpModelProcessor();
}
