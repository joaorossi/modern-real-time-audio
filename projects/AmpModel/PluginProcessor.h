#pragma once

#include <JuceHeader.h>
#include "Gru.h"
#include "AmpGruParameters.h"

namespace Param
{
    namespace ID
    {
        static const juce::String Volume { "volume" };
        static const juce::String Tone { "tone" };
    }

    namespace Name
    {
        static const juce::String Volume { "Volume" };
        static const juce::String Tone { "Tone" };
    }
}

class AmpModelProcessor : public juce::AudioProcessor
{
public:
    AmpModelProcessor();
    ~AmpModelProcessor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    mrta::ParameterManager& getParameterManager() { return parameterManager; }

    //==============================================================================
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;
    const juce::String getName() const override;
    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const juce::String getProgramName (int index) override;
    void changeProgramName (int index, const juce::String& newName) override;
    //==============================================================================

private:
    mrta::ParameterManager parameterManager;
    juce::SmoothedValue<float> volume;
    juce::SmoothedValue<float> tone;

    juce::AudioBuffer<float> nnInputBuffer;
    juce::AudioBuffer<float> nnOutputBuffer;

    static const size_t INPUT_SIZE = 3u;
    static const size_t OUTPUT_SIZE = 1u;
    static const size_t HIDDEN_SIZE = 16u;

    Gru<INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZE> gru[2];

    AmpGruParameters gruParameters;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AmpModelProcessor)
};
