#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"

class AmpModelProcessorEditor : public juce::AudioProcessorEditor
{
public:
    AmpModelProcessorEditor(AmpModelProcessor&);
    ~AmpModelProcessorEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;

private:
    AmpModelProcessor& audioProcessor;
    mrta::GenericParameterEditor genericParameterEditor;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AmpModelProcessorEditor)
};
